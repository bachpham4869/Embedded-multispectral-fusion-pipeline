import cv2 as cv
import sys
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from smartbinocular.nir_pipeline import nir_anti_glare_bgr

manifest = Path("data/eval/nir_val/manifest_v2.csv")
glare_path = None
import csv
with open(manifest, newline="") as f:
    for row in csv.DictReader(f):
        if row["path"].endswith("input_000052.png"):
            glare_path = row["path"]
            break

bgr = cv.imread(glare_path, cv.IMREAD_COLOR)
bgr_small = cv.resize(bgr, (320, 240))
out = nir_anti_glare_bgr(bgr_small)

diff = out.astype(np.int16) - bgr_small.astype(np.int16)

# Find max positive and negative diff
max_pos = np.max(diff)
min_neg = np.min(diff)

# Count how many pixels changed significantly
num_changed = np.sum(np.abs(diff) > 5)

print(f"Max positive diff (boost): {max_pos}")
print(f"Max negative diff (darken): {min_neg}")
print(f"Num pixels changed > 5: {num_changed} out of {320*240*3}")

# Find standard deviation
print(f"Std diff: {np.std(diff)}")

