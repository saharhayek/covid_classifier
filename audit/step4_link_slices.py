"""Stage A, step 4: rebuild scans by linking each slice to its next slice.

Step 3 showed that the files cycle through patients, but with a period that
changes along the numbering. So instead of assuming one period, we look for
each slice's own successor: the most similar slice among the next MAX_LAG
files of the same class and image size. Links are accepted strongest first,
and each slice can have at most one successor and one predecessor. Following
the links gives chains, and each chain is one reconstructed scan.

This also handles the data that is already in scan order (successor at lag 1),
so it replaces the scan cuts from step 2.

Needs the output of step 1. Run from the project root:
    python audit/step4_link_slices.py

Outputs in outputs/audit/:
    scans_linked.csv           one row per slice with scan_id and position in the scan
    step4_link_sims.png        how similar each slice is to its best successor
    step4_chains.png           random reconstructed scans, to check by eye
"""
import csv
import random
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

AUDIT_DIR = Path("outputs") / "audit"
ORIGINAL = Path("dataset") / "Original CT Scans"
CLASSES = ["nCT", "NiCT", "pCT"]

# A link needs at least this similarity. Check step4_link_sims.png: the true
# links should form a peak near 1, separate from the rest.
LINK_THRESHOLD = 0.90
MAX_LAG = 600       # how far ahead to look for the next slice
CANDIDATES = 3      # candidate successors kept per slice
MIN_CHAIN_PLOT = 20
N_EXAMPLES = 6
SLICES_SHOWN = 8
random.seed(0)

rows = list(csv.DictReader(open(AUDIT_DIR / "inventory.csv")))
thumbs = np.load(AUDIT_DIR / "thumbs.npy")
assert len(rows) == len(thumbs), "inventory.csv and thumbs.npy do not match; rerun step 1"

x = thumbs.reshape(len(thumbs), -1).astype(np.float32)
x -= x.mean(axis=1, keepdims=True)
x /= np.linalg.norm(x, axis=1, keepdims=True) + 1e-6

out_rows = []
best_sims = {}
chains_by_class = {}
for cls in CLASSES:
    idx = [i for i, r in enumerate(rows) if r["cls"] == cls]
    idx.sort(key=lambda i: int(rows[i]["num"]))
    feats = x[idx]
    widths = np.array([rows[i]["width"] for i in idx])
    n = len(idx)

    # Candidate links: for each slice, the most similar slices ahead of it
    edges = []
    best = np.full(n, np.nan, dtype=np.float32)
    for i in range(n - 1):
        hi = min(n, i + MAX_LAG + 1)
        s = feats[i + 1:hi] @ feats[i]
        s[widths[i + 1:hi] != widths[i]] = -1.0          # never link across image sizes
        top = np.argsort(s)[::-1][:CANDIDATES]
        best[i] = s[top[0]]
        for t in top:
            if s[t] >= LINK_THRESHOLD:
                edges.append((float(s[t]), i, i + 1 + int(t)))
    best_sims[cls] = best[~np.isnan(best)]

    # Accept the strongest links first; one successor and one predecessor per slice
    edges.sort(reverse=True)
    succ = np.full(n, -1)
    pred = np.full(n, -1)
    link_sim = np.zeros(n, dtype=np.float32)
    for s, i, j in edges:
        if succ[i] == -1 and pred[j] == -1:
            succ[i], pred[j], link_sim[j] = j, i, s

    # Follow the links from every slice that has no predecessor
    chains = []
    for start in np.where(pred == -1)[0]:
        chain = [int(start)]
        while succ[chain[-1]] != -1:
            chain.append(int(succ[chain[-1]]))
        chains.append(chain)
    assert sum(len(c) for c in chains) == n
    chains.sort(key=lambda c: int(rows[idx[c[0]]]["num"]))
    chains_by_class[cls] = (idx, chains)

    for k, chain in enumerate(chains, start=1):
        for pos, p in enumerate(chain):
            r = rows[idx[p]]
            out_rows.append({
                "cls": cls,
                "num": r["num"],
                "file": r["file"],
                "width": r["width"],
                "scan_id": f"{cls}_{k:04d}",
                "pos_in_scan": pos,
                "scan_len": len(chain),
                "link_sim": f"{link_sim[p]:.4f}" if pos > 0 else "",
            })

    lengths = np.array([len(c) for c in chains])
    gaps = Counter(int(succ[i] - i) for i in range(n) if succ[i] != -1)
    linked = (succ != -1).sum()
    print(f"\n{cls}: {n} slices -> {len(chains)} scans")
    print(f"  slices per scan: median {int(np.median(lengths))}, max {lengths.max()} | "
          f"scans of 1 slice: {(lengths == 1).sum()}, of 2-4: {((lengths >= 2) & (lengths <= 4)).sum()}, "
          f"of 5+: {(lengths >= 5).sum()} (holding {lengths[lengths >= 5].sum()} slices)")
    print(f"  slices with a successor: {linked}/{n - 1}")
    print("  most common jumps to the next slice: "
          + ", ".join(f"{g} ({c})" for g, c in gaps.most_common(10)))

with open(AUDIT_DIR / "scans_linked.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=out_rows[0].keys())
    writer.writeheader()
    writer.writerows(out_rows)

# Plot 1: similarity of each slice to its best successor
fig, axes = plt.subplots(1, len(CLASSES), figsize=(15, 4), sharey=True)
for ax, cls in zip(axes, CLASSES):
    ax.hist(best_sims[cls], bins=100, range=(0, 1), color="tab:blue")
    ax.axvline(LINK_THRESHOLD, color="tab:red", ls="--", lw=1, label=f"threshold {LINK_THRESHOLD}")
    ax.set_yscale("log")
    ax.set_title(cls)
    ax.set_xlabel("similarity to best successor")
    ax.legend(fontsize=8)
axes[0].set_ylabel("slices")
fig.tight_layout()
fig.savefig(AUDIT_DIR / "step4_link_sims.png", dpi=120)
plt.close(fig)

# Plot 2: random long chains, with evenly spaced slices from first to last
candidates = [(cls, c) for cls in CLASSES for c in chains_by_class[cls][1]
              if len(c) >= MIN_CHAIN_PLOT]
examples = random.sample(candidates, min(N_EXAMPLES, len(candidates)))
fig, axes = plt.subplots(len(examples), SLICES_SHOWN,
                         figsize=(1.6 * SLICES_SHOWN, 1.9 * len(examples)), squeeze=False)
for row_axes, (cls, chain) in zip(axes, examples):
    idx = chains_by_class[cls][0]
    picks = np.linspace(0, len(chain) - 1, SLICES_SHOWN).round().astype(int)
    for ax, p in zip(row_axes, picks):
        r = rows[idx[chain[p]]]
        with Image.open(ORIGINAL / cls / r["file"]) as img:
            ax.imshow(img.convert("L").resize((128, 128)), cmap="gray")
        ax.set_title(f"{r['file']}\n{p + 1}/{len(chain)}", fontsize=6)
        ax.axis("off")
fig.suptitle("Each row should be ONE patient, moving through the body", fontsize=9)
fig.tight_layout()
fig.savefig(AUDIT_DIR / "step4_chains.png", dpi=120)
plt.close(fig)

print(f"\nSaved {AUDIT_DIR / 'scans_linked.csv'}, step4_link_sims.png and step4_chains.png")
