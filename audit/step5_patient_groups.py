"""Stage A, step 5: join the reconstructed scans into patient groups.

Step 4 rebuilt scans within each class, but one patient is still split into
pieces: slices above and below the lungs are in NiCT, and a COVID patient's
normal-looking slices are in nCT between their pCT slices. Pieces of the same
patient contain neighbouring slices of the same body, so some slice in one
piece is almost identical to some slice in the other.

For every pair of scans with the same image size (across all three classes)
we take the highest similarity between any slice of one and any slice of the
other. Pairs at or above GROUP_THRESHOLD are joined, and joined scans form a
patient group.

For the patient split, merging two patients into one group is harmless (they
just stay on the same side), but leaving one patient in two groups leaks data.
So a slightly too-low threshold is safer than a too-high one, as long as the
groups do not collapse into one giant group.

Needs the outputs of steps 1 and 4. Run from the project root:
    python audit/step5_patient_groups.py

Outputs in outputs/audit/:
    patient_groups.csv         one row per slice with its scan_id and group_id
    step5_scan_links.png       each scan's best similarity to any other scan
    step5_groups.png           random groups, one slice per scan, to check by eye
"""
import csv
import random
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

AUDIT_DIR = Path("outputs") / "audit"
ORIGINAL = Path("dataset") / "Original CT Scans"

GROUP_THRESHOLD = 0.96
SWEEP = [0.90, 0.93, 0.95, 0.96, 0.97, 0.98, 0.99]
CHUNK = 1000          # rows per block when comparing slices (limits memory use)
N_EXAMPLES = 6
MAX_SCANS_SHOWN = 8
random.seed(0)

inventory = list(csv.DictReader(open(AUDIT_DIR / "inventory.csv")))
thumbs = np.load(AUDIT_DIR / "thumbs.npy")
assert len(inventory) == len(thumbs), "inventory.csv and thumbs.npy do not match; rerun step 1"
thumb_of = {(r["cls"], r["file"]): i for i, r in enumerate(inventory)}

x = thumbs.reshape(len(thumbs), -1).astype(np.float32)
x -= x.mean(axis=1, keepdims=True)
x /= np.linalg.norm(x, axis=1, keepdims=True) + 1e-6

slices = list(csv.DictReader(open(AUDIT_DIR / "scans_linked.csv")))
scan_ids = sorted({s["scan_id"] for s in slices})
scan_index = {sid: k for k, sid in enumerate(scan_ids)}
n_scans = len(scan_ids)

scan_of = np.array([scan_index[s["scan_id"]] for s in slices])
feat_idx = np.array([thumb_of[(s["cls"], s["file"])] for s in slices])
width = np.array([s["width"] for s in slices])

scan_cls = {}
scan_len = Counter()
scan_members = defaultdict(list)          # scan -> slice rows, in scan order
for k, s in enumerate(slices):
    scan_cls[scan_of[k]] = s["cls"]
    scan_len[scan_of[k]] += 1
    scan_members[scan_of[k]].append(k)

# best[a, b] = highest similarity between any slice of scan a and any slice of scan b
best = np.full((n_scans, n_scans), -1.0, dtype=np.float32)
for w in sorted(set(width)):
    sel = np.where(width == w)[0]
    sel = sel[np.argsort(scan_of[sel], kind="stable")]       # group columns by scan
    feats = x[feat_idx[sel]]
    owner = scan_of[sel]
    col_scans, col_starts = np.unique(owner, return_index=True)
    print(f"Comparing {len(sel)} slices of width {w} ({len(col_scans)} scans)")
    for a in range(0, len(sel), CHUNK):
        sims = feats[a:a + CHUNK] @ feats.T
        per_col = np.maximum.reduceat(sims, col_starts, axis=1)
        row_scans, row_starts = np.unique(owner[a:a + CHUNK], return_index=True)
        per_pair = np.maximum.reduceat(per_col, row_starts, axis=0)
        block = np.ix_(row_scans, col_scans)
        best[block] = np.maximum(best[block], per_pair)
best = np.maximum(best, best.T)
np.fill_diagonal(best, -1.0)


def find_groups(threshold):
    parent = list(range(n_scans))

    def root(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for a, b in zip(*np.where(np.triu(best >= threshold, k=1))):
        parent[root(a)] = root(b)
    roots = [root(a) for a in range(n_scans)]
    groups = defaultdict(list)
    for a, r in enumerate(roots):
        groups[r].append(a)
    return list(groups.values())


def describe(groups):
    sizes = sorted((sum(scan_len[a] for a in g) for g in groups), reverse=True)
    with_pct = sum(any(scan_cls[a] == "pCT" for a in g) for g in groups)
    return len(groups), with_pct, sizes


print("\nGroups per threshold (target: about 104 patients, 61 with COVID):")
print(f"  {'threshold':>9s} {'groups':>7s} {'with pCT':>9s} {'without':>8s} "
      f"{'largest group (slices)':>23s} {'groups < 20 slices':>19s}")
for t in SWEEP:
    n, with_pct, sizes = describe(find_groups(t))
    mark = "  <- used" if t == GROUP_THRESHOLD else ""
    small = sum(s < 20 for s in sizes)
    print(f"  {t:9.2f} {n:7d} {with_pct:9d} {n - with_pct:8d} {sizes[0]:23d} {small:19d}{mark}")

groups = find_groups(GROUP_THRESHOLD)
groups.sort(key=lambda g: -sum(scan_len[a] for a in g))
group_of = {}
for k, g in enumerate(groups, start=1):
    for a in g:
        group_of[a] = f"G{k:03d}"

n, with_pct, sizes = describe(groups)
print(f"\nWith threshold {GROUP_THRESHOLD}: {n} groups, {with_pct} contain pCT slices")
print(f"  slices per group: median {int(np.median(sizes))}, largest {sizes[:5]}")
print("  scans per group:", Counter(len(g) for g in groups).most_common(10))
print("  class mix of groups:",
      Counter("+".join(sorted({scan_cls[a] for a in g})) for g in groups).most_common())

with open(AUDIT_DIR / "patient_groups.csv", "w", newline="") as f:
    fields = ["cls", "num", "file", "width", "scan_id", "group_id"]
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    for k, s in enumerate(slices):
        writer.writerow({**{c: s[c] for c in fields[:5]}, "group_id": group_of[scan_of[k]]})

# Plot 1: for each scan, its best similarity to any other scan
fig, ax = plt.subplots(figsize=(9, 4))
ax.hist(best.max(axis=1), bins=100, range=(0, 1), color="tab:blue")
ax.axvline(GROUP_THRESHOLD, color="tab:red", ls="--", lw=1, label=f"threshold {GROUP_THRESHOLD}")
ax.set_yscale("log")
ax.set_xlabel("best similarity between a scan and any other scan")
ax.set_ylabel("scans")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(AUDIT_DIR / "step5_scan_links.png", dpi=120)
plt.close(fig)

# Plot 2: random groups with several scans; middle slice of each scan
multi = [g for g in groups if len(g) >= 3]
examples = random.sample(multi, min(N_EXAMPLES, len(multi)))
fig, axes = plt.subplots(len(examples), MAX_SCANS_SHOWN,
                         figsize=(1.6 * MAX_SCANS_SHOWN, 1.9 * len(examples)), squeeze=False)
for row_axes, g in zip(axes, examples):
    shown = sorted(g, key=lambda a: -scan_len[a])[:MAX_SCANS_SHOWN]
    for ax in row_axes:
        ax.axis("off")
    for ax, a in zip(row_axes, shown):
        members = scan_members[a]
        s = slices[members[len(members) // 2]]
        with Image.open(ORIGINAL / s["cls"] / s["file"]) as img:
            ax.imshow(img.convert("L").resize((128, 128)), cmap="gray")
        ax.set_title(f"{group_of[a]} {scan_ids[a]}\n{s['file']} ({scan_len[a]} sl.)", fontsize=6)
fig.suptitle("Each row should be ONE patient (same body and table, different heights)",
             fontsize=9)
fig.tight_layout()
fig.savefig(AUDIT_DIR / "step5_groups.png", dpi=120)
plt.close(fig)

print(f"\nSaved {AUDIT_DIR / 'patient_groups.csv'}, step5_scan_links.png and step5_groups.png")
