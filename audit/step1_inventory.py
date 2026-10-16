"""Stage A, step 1: inventory every slice in the dataset.

For each image we record its class, its number in the filename, size, colour
mode, an exact content hash and a small grayscale thumbnail. The thumbnails are
used in later steps to find near-duplicates and to recover scan boundaries.

Run from the project root:
    python audit/step1_inventory.py
"""
import csv
import hashlib
import re
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

DATA_ROOT = Path("dataset")
ORIGINAL = DATA_ROOT / "Original CT Scans"
PREPROCESSED = DATA_ROOT / "Preprocessed CT scans"
OUT_DIR = Path("outputs") / "audit"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CLASSES = ["nCT", "NiCT", "pCT"]
THUMB = 32


def slice_number(path):
    return int(re.search(r"(\d+)$", path.stem).group(1))


def list_files(root):
    files = {}
    for cls in CLASSES:
        for p in (root / cls).iterdir():
            if p.is_file():
                files[(cls, p.name)] = p
    return files


original = list_files(ORIGINAL)
preprocessed = list_files(PREPROCESSED)

print("Files per class (original / preprocessed):")
for cls in CLASSES:
    n_o = sum(1 for c, _ in original if c == cls)
    n_p = sum(1 for c, _ in preprocessed if c == cls)
    print(f"  {cls:5s} {n_o:6d} / {n_p:6d}")
only_o = set(original) - set(preprocessed)
only_p = set(preprocessed) - set(original)
print(f"Only in original: {len(only_o)} | only in preprocessed: {len(only_p)}")

# Are the numbers in each class a gap-free 1..N sequence?
for cls in CLASSES:
    nums = sorted(slice_number(p) for (c, _), p in original.items() if c == cls)
    missing = sorted(set(range(1, nums[-1] + 1)) - set(nums))
    print(f"  {cls}: numbers {nums[0]}..{nums[-1]}, missing {len(missing)}")

rows = []
thumbs = []
keys = sorted(original, key=lambda k: (k[0], slice_number(original[k])))
for i, key in enumerate(keys):
    path = original[key]
    raw = path.read_bytes()
    with Image.open(path) as img:
        size, mode = img.size, img.mode
        thumb = img.convert("L").resize((THUMB, THUMB), Image.BILINEAR)
    thumbs.append(np.asarray(thumb, dtype=np.uint8))
    rows.append({
        "cls": key[0],
        "num": slice_number(path),
        "file": key[1],
        "width": size[0],
        "height": size[1],
        "mode": mode,
        "bytes": len(raw),
        "md5": hashlib.md5(raw).hexdigest(),
    })
    if (i + 1) % 2000 == 0:
        print(f"  read {i + 1}/{len(keys)}")

with open(OUT_DIR / "inventory.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)
np.save(OUT_DIR / "thumbs.npy", np.stack(thumbs))

print("\nImage sizes:", Counter((r["width"], r["height"]) for r in rows).most_common(8))
print("Colour modes:", Counter(r["mode"] for r in rows))

dupes = Counter(r["md5"] for r in rows)
dupe_groups = [h for h, n in dupes.items() if n > 1]
dupe_files = sum(dupes[h] for h in dupe_groups)
cross_class = sum(
    1 for h in dupe_groups if len({r["cls"] for r in rows if r["md5"] == h}) > 1
)
print(f"Exact duplicates: {len(dupe_groups)} groups covering {dupe_files} files "
      f"({cross_class} groups span more than one class)")
print(f"\nSaved {OUT_DIR / 'inventory.csv'} and {OUT_DIR / 'thumbs.npy'}")