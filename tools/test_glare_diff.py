import cv2 as cv
import sys
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from smartbinocular.nir_pipeline import nir_glare_eval, nir_anti_glare_bgr, _nir_gray_for_stats, _nir_glare_metrics_from_gray

manifest = Path("data/eval/nir_val/manifest_v2.csv")
import csv
with open(manifest, newline="") as f:
    for row in csv.DictReader(f):
        if row["env_class"] in ("glare", "backlight"):
            bgr = cv.imread(row["path"], cv.IMREAD_COLOR)
            out = nir_anti_glare_bgr(bgr)
            is_diff = not np.array_equal(bgr, out)
            mean_b = float(np.mean(bgr))
            g = _nir_gray_for_stats(bgr)
            need, hud, p_hi, p99, mx = _nir_glare_metrics_from_gray(g, use_fast=True)
            print(f"{Path(row['path']).name}: mean={mean_b:.1f}, p_hi={p_hi:.1f}, p99={p99:.1f}, mx={mx:.1f} -> need={need}, applied={is_diff}")

