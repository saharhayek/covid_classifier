"""Stage A, step 2: recover scan boundaries from the slice numbering.

Within a class, consecutive file numbers look like consecutive slices of the
same scan. Neighbouring slices are very similar, so similarity between slice n
and slice n+1 should stay high inside a scan and drop where a new scan starts.
A change of image size between neighbours is treated as a certain boundary,
because one scan always has one image size.

Needs the output of step 1. Run from the project root:
    python audit/step2_scan_boundaries.py

Outputs in outputs/audit/:
    scans.csv                  one row per slice with its recovered scan_id
    step2_similarity.png       similarity curve per class, with the threshold
    step2_boundaries.png       example boundaries to check by eye
"""
import csv
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

AUDIT_DIR = Path("outputs") / "audit"
ORIGINAL = Path("dataset") / "Original CT Scans"
CLASSES = ["nCT", "NiCT", "pCT"]

# Similarity below this between two neighbouring slices starts a new scan.
# Look at step2_similarity.png and the sweep table, then adjust if needed.
THRESHOLD = 0.80
SWEEP = [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
N_EXAMPLES = 8
random.seed(0)

rows = list(csv.DictReader(open(AUDIT_DIR / "inventory.csv")))
thumbs = np.load(AUDIT_DIR / "thumbs.npy")
assert len(rows) == len(thumbs), "inventory.csv and thumbs.npy do not match; rerun step 1"

# Normalise each thumbnail (zero mean, unit length) so a dot product is a correlation
x = thumbs.reshape(len(thumbs), -1).astype(np.float32)
x -= x.mean(axis=1, keepdims=True)
x /= np.linalg.norm(x, axis=1, keepdims=True) + 1e-6

per_class = {}
for cls in CLASSES:
    idx = [i for i, r in enumerate(rows) if r["cls"] == cls]
    idx.sort(key=lambda i: int(rows[i]["num"]))
    sim = np.ones(len(idx), dtype=np.float32)          # first slice has no predecessor
    sim[1:] = np.einsum("ij,ij->i", x[idx[:-1]], x[idx[1:]])
    size = [rows[i]["width"] for i in idx]
    size_change = np.array([False] + [a != b for a, b in zip(size, size[1:])])
    per_class[cls] = (idx, sim, size_change)


def count_scans(sim, size_change, t):
    starts = (sim < t) | size_change
    starts[0] = True
    return int(starts.sum())


print("Number of scans per threshold:")
print(f"  {'threshold':>9s} " + " ".join(f"{c:>6s}" for c in CLASSES) + "   total")
for t in SWEEP:
    counts = [count_scans(per_class[c][1], per_class[c][2], t) for c in CLASSES]
    mark = "  <- used" if t == THRESHOLD else ""
    print(f"  {t:9.2f} " + " ".join(f"{n:6d}" for n in counts) + f"  {sum(counts):6d}{mark}")

# Assign scan ids with the chosen threshold
out_rows = []
boundaries = []                                         # (cls, position in class order)
summary = {}
for cls in CLASSES:
    idx, sim, size_change = per_class[cls]
    starts = (sim < THRESHOLD) | size_change
    starts[0] = True
    scan_no = np.cumsum(starts)
    lengths = np.bincount(scan_no)[1:]
    summary[cls] = lengths
    for pos, i in enumerate(idx):
        r = rows[i]
        out_rows.append({
            "cls": cls,
            "num": r["num"],
            "file": r["file"],
            "width": r["width"],
            "scan_id": f"{cls}_{scan_no[pos]:03d}",
            "sim_to_prev": f"{sim[pos]:.4f}",
            "scan_start": int(starts[pos]),
            "size_change": int(size_change[pos]),
        })
        if pos > 0 and starts[pos]:
            boundaries.append((cls, pos))

with open(AUDIT_DIR / "scans.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=out_rows[0].keys())
    writer.writeheader()
    writer.writerows(out_rows)

print(f"\nWith threshold {THRESHOLD}:")
for cls in CLASSES:
    L = summary[cls]
    print(f"  {cls:5s} {len(L):4d} scans | slices per scan: median {int(np.median(L))}, "
          f"min {L.min()}, max {L.max()} | single-slice scans: {(L == 1).sum()}")

# Plot 1: similarity along the numbering, plus a histogram per class
fig, axes = plt.subplots(len(CLASSES), 2, figsize=(16, 9),
                         gridspec_kw={"width_ratios": [4, 1]})
for row_axes, cls in zip(axes, CLASSES):
    idx, sim, size_change = per_class[cls]
    nums = [int(rows[i]["num"]) for i in idx]
    ax, hax = row_axes
    ax.plot(nums, sim, lw=0.4, color="tab:blue")
    ax.axhline(THRESHOLD, color="tab:red", lw=0.8, ls="--", label=f"threshold {THRESHOLD}")
    for n, ch in zip(nums, size_change):
        if ch:
            ax.axvline(n, color="tab:orange", lw=0.8, alpha=0.7)
    ax.set_ylim(-0.2, 1.02)
    ax.set_ylabel(f"{cls}\nsimilarity to previous")
    ax.legend(loc="lower left", fontsize=8)
    hax.hist(sim[1:], bins=60, range=(-0.2, 1), orientation="horizontal", color="tab:blue")
    hax.axhline(THRESHOLD, color="tab:red", lw=0.8, ls="--")
    hax.set_xscale("log")
    hax.set_ylim(-0.2, 1.02)
axes[-1][0].set_xlabel("file number (orange lines = image size changes)")
fig.tight_layout()
fig.savefig(AUDIT_DIR / "step2_similarity.png", dpi=120)
plt.close(fig)

# Plot 2: random boundaries, showing two slices before and two after each cut
examples = random.sample(boundaries, min(N_EXAMPLES, len(boundaries)))
fig, axes = plt.subplots(len(examples), 4, figsize=(8, 2.1 * len(examples)), squeeze=False)
for row_axes, (cls, pos) in zip(axes, examples):
    idx, sim, _ = per_class[cls]
    for ax, p in zip(row_axes, range(pos - 2, pos + 2)):
        ax.axis("off")
        if 0 <= p < len(idx):
            r = rows[idx[p]]
            with Image.open(ORIGINAL / cls / r["file"]) as img:
                ax.imshow(img.convert("L").resize((128, 128)), cmap="gray")
            side = "before" if p < pos else "after"
            ax.set_title(f"{r['file']} ({side})\nsim {sim[p]:.2f}", fontsize=7)
fig.suptitle("Detected scan boundaries: the cut is between columns 2 and 3", fontsize=9)
fig.tight_layout()
fig.savefig(AUDIT_DIR / "step2_boundaries.png", dpi=120)
plt.close(fig)

print(f"\nSaved {AUDIT_DIR / 'scans.csv'}, step2_similarity.png and step2_boundaries.png")
