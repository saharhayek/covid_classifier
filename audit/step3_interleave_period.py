"""Stage A, step 3: find the interleaving period in the 512 px block.

Step 2 showed that in the 512 px images, neighbouring file numbers are
different patients at about the same height in the chest. If the files cycle
through P patients, then file n and file n+P are the same patient on
neighbouring slices, so similarity should peak at lag P (and at 2P, 3P, ...).

This script measures the median similarity at every lag from 1 to MAX_LAG,
for the whole 512 px block of each class and for windows of WINDOW files,
because the pattern in step 2 changes part-way through.

Needs the output of step 1. Run from the project root:
    python audit/step3_interleave_period.py

Outputs in outputs/audit/:
    step3_lag_profile.png      median similarity per lag, one curve per class
    step3_windows.csv          best lag in each window
"""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

AUDIT_DIR = Path("outputs") / "audit"
CLASSES = ["nCT", "NiCT", "pCT"]
BLOCK_WIDTH = "512"   # image size of the interleaved block
MAX_LAG = 400
WINDOW = 500
TOP_K = 5             # how many candidate lags to print

rows = list(csv.DictReader(open(AUDIT_DIR / "inventory.csv")))
thumbs = np.load(AUDIT_DIR / "thumbs.npy")
assert len(rows) == len(thumbs), "inventory.csv and thumbs.npy do not match; rerun step 1"

x = thumbs.reshape(len(thumbs), -1).astype(np.float32)
x -= x.mean(axis=1, keepdims=True)
x /= np.linalg.norm(x, axis=1, keepdims=True) + 1e-6


def lag_profile(feats, max_lag):
    """Median correlation between item i and item i+lag, for lag = 1..max_lag."""
    prof = np.full(max_lag + 1, np.nan, dtype=np.float32)
    for lag in range(1, min(max_lag, len(feats) - 1) + 1):
        prof[lag] = np.median(np.einsum("ij,ij->i", feats[:-lag], feats[lag:]))
    return prof


def top_lags(prof, k):
    order = np.argsort(np.nan_to_num(prof[1:], nan=-1))[::-1] + 1
    return [(int(lag), float(prof[lag])) for lag in order[:k]]


fig, axes = plt.subplots(len(CLASSES), 1, figsize=(14, 9), sharex=True)
window_rows = []
for ax, cls in zip(axes, CLASSES):
    idx = [i for i, r in enumerate(rows) if r["cls"] == cls]
    idx.sort(key=lambda i: int(rows[i]["num"]))
    # Keep only the 512 px slices, in file-number order
    idx = [i for i in idx if rows[i]["width"] == BLOCK_WIDTH]
    nums = [int(rows[i]["num"]) for i in idx]
    feats = x[idx]
    print(f"\n{cls}: {len(idx)} images at {BLOCK_WIDTH} px, numbers {nums[0]}..{nums[-1]}")

    prof = lag_profile(feats, MAX_LAG)
    print(f"  whole block, best lags: "
          + ", ".join(f"{lag} ({s:.2f})" for lag, s in top_lags(prof, TOP_K)))
    ax.plot(range(MAX_LAG + 1), prof, lw=1, color="black", label="whole block")

    for start in range(0, len(idx), WINDOW):
        w = feats[start:start + WINDOW]
        if len(w) < 50:
            continue
        wprof = lag_profile(w, min(MAX_LAG, len(w) // 2))
        best = top_lags(wprof, TOP_K)
        lo, hi = nums[start], nums[min(start + WINDOW, len(idx)) - 1]
        print(f"  files {lo:5d}-{hi:5d}: best lags "
              + ", ".join(f"{lag} ({s:.2f})" for lag, s in best)
              + f" | lag 1: {wprof[1]:.2f}")
        window_rows.append({
            "cls": cls, "first_num": lo, "last_num": hi,
            "best_lag": best[0][0], "best_sim": f"{best[0][1]:.3f}",
            "lag1_sim": f"{wprof[1]:.3f}",
            "next_lags": " ".join(str(lag) for lag, _ in best[1:]),
        })
        ax.plot(range(len(wprof)), wprof, lw=0.4, alpha=0.5)

    ax.set_ylabel(f"{cls}\nmedian similarity")
    ax.set_ylim(0, 1)
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.3)
axes[-1].set_xlabel("lag (difference in file number); thin lines = windows of "
                    f"{WINDOW} files")
fig.tight_layout()
fig.savefig(AUDIT_DIR / "step3_lag_profile.png", dpi=120)
plt.close(fig)

with open(AUDIT_DIR / "step3_windows.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=window_rows[0].keys())
    writer.writeheader()
    writer.writerows(window_rows)

print(f"\nSaved {AUDIT_DIR / 'step3_lag_profile.png'} and step3_windows.csv")
