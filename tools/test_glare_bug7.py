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

# Process at FULL resolution
out_full = nir_anti_glare_bgr(bgr)
diff_full = out_full.astype(np.int16) - bgr.astype(np.int16)

print(f"FULL - Max positive diff (boost): {np.max(diff_full)}")
print(f"FULL - Max negative diff (darken): {np.min(diff_full)}")
print(f"FULL - Num pixels changed > 5: {np.sum(np.abs(diff_full) > 5)} out of {bgr.shape[0]*bgr.shape[1]*3}")
print(f"FULL - Std diff: {np.std(diff_full)}")

