import cv2 as cv
import sys
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from smartbinocular.nir_pipeline import nir_anti_glare_bgr

manifest = Path("data/eval/nir_val/manifest_v2.csv")

def check_img(name):
    glare_path = None
    import csv
    with open(manifest, newline="") as f:
        for row in csv.DictReader(f):
            if row["path"].endswith(name):
                glare_path = row["path"]
                break
    bgr = cv.imread(glare_path, cv.IMREAD_COLOR)
    out_full = nir_anti_glare_bgr(bgr)
    diff_full = out_full.astype(np.int16) - bgr.astype(np.int16)
    print(f"{name} - Mean: {np.mean(bgr):.1f}, Max boost: {np.max(diff_full)}, Num changed > 5: {np.sum(np.abs(diff_full) > 5)} / {bgr.shape[0]*bgr.shape[1]*3}")

check_img("input_000040.png")
check_img("input_000027.png")
check_img("input_000052.png")
