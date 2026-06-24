import cv2 as cv
import sys
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from smartbinocular.nir_pipeline import nir_nir_night_clahe, nir_anti_glare_bgr

manifest = Path("data/eval/nir_val/manifest_v2.csv")
glare_paths = []
import csv
with open(manifest, newline="") as f:
    for row in csv.DictReader(f):
        if row["env_class"] in ("glare", "backlight"):
            glare_paths.append(row["path"])

for name in ["input_000027.png", "input_000040.png", "input_000052.png", "0054.JPG", "input_000062.png"]:
    path = next((p for p in glare_paths if p.endswith(name)), None)
    if not path:
        continue
    bgr = cv.resize(cv.imread(path, cv.IMREAD_COLOR), (320, 240))
    clahe = nir_nir_night_clahe(bgr)
    tone = nir_anti_glare_bgr(bgr)
    
    # We want CLAHE to look VERY DIFFERENT from Tone Map (which proves the point).
    # But we also want the RAW image to look like a glare image.
    diff_clahe_raw = np.mean(np.abs(clahe.astype(np.int16) - bgr.astype(np.int16)))
    diff_tone_raw = np.mean(np.abs(tone.astype(np.int16) - bgr.astype(np.int16)))
    
    print(f"{name}: Mean={np.mean(bgr):.1f}, Diff(CLAHE-Raw)={diff_clahe_raw:.1f}, Diff(Tone-Raw)={diff_tone_raw:.1f}")

