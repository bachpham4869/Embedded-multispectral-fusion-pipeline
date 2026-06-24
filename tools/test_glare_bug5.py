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

diff = np.abs(bgr_small.astype(np.int16) - out.astype(np.int16))
print(f"Max diff: {np.max(diff)}")
print(f"Mean diff: {np.mean(diff)}")

