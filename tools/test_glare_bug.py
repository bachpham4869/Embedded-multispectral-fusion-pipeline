import cv2 as cv
import sys
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from smartbinocular.nir_pipeline import nir_a1_lite_tone_map_bgr

manifest = Path("data/eval/nir_val/manifest_v2.csv")
glare_path = None
import csv
with open(manifest, newline="") as f:
    for row in csv.DictReader(f):
        if row["path"].endswith("input_000027.png"):
            glare_path = row["path"]
            break

bgr = cv.imread(glare_path, cv.IMREAD_COLOR)
out = nir_a1_lite_tone_map_bgr(bgr)

print("BGR min/max:", np.min(bgr), np.max(bgr))
print("OUT min/max:", np.min(out), np.max(out))

gray = bgr[:, :, 1].astype(np.float32)
g0 = np.clip(gray / 255.0, 1e-6, 1.0)
g = np.power(g0, 0.72)
g = np.where(g > 0.92, 0.92 + (g - 0.92) * 0.38, g)
ratio = np.clip(g / g0, 0.0, 1.55)
span = max(0.48 - 0.17, 1e-6)
w = np.clip((g0 - 0.17) / span, 0.0, 1.0)
ratio_eff = 1.0 + w * (ratio - 1.0)

print("ratio_eff min/max:", np.min(ratio_eff), np.max(ratio_eff))
prod = bgr.astype(np.float32) * ratio_eff[..., None]
print("prod min/max:", np.min(prod), np.max(prod))

