import cv2 as cv
import sys
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from smartbinocular.nir_pipeline import nir_nir_night_clahe

bgr = cv.imread("data/eval/nir_val/normal_day/input_000000.png", cv.IMREAD_COLOR) # Or any glare image
manifest = Path("data/eval/nir_val/manifest_v2.csv")
glare_path = None
import csv
with open(manifest, newline="") as f:
    for row in csv.DictReader(f):
        if row["path"].endswith("input_000091.png"):
            glare_path = row["path"]
            break

bgr = cv.imread(glare_path, cv.IMREAD_COLOR)
out_b = nir_nir_night_clahe(bgr)

diff = np.abs(bgr.astype(np.int16) - out_b.astype(np.int16))
print(f"CLAHE (Bucket B) Max diff: {np.max(diff)}")
print(f"CLAHE (Bucket B) Mean diff: {np.mean(diff)}")

