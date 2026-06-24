import cv2 as cv
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from smartbinocular.nir_pipeline import nir_anti_glare_bgr

manifest = Path("data/eval/nir_val/manifest_v2.csv")
glare_rows = []
import csv
with open(manifest, newline="") as f:
    for row in csv.DictReader(f):
        if row["env_class"] in ("glare", "backlight"):
            glare_rows.append(row["path"])

glare_rows = glare_rows[:3]

for path in glare_rows:
    bgr = cv.imread(path, cv.IMREAD_COLOR)
    out = nir_anti_glare_bgr(bgr)
    
    print(f"{Path(path).name}:")
    print(f"  bgr shape={bgr.shape}, dtype={bgr.dtype}, min={np.min(bgr)}, max={np.max(bgr)}")
    print(f"  out shape={out.shape}, dtype={out.dtype}, min={np.min(out)}, max={np.max(out)}")
    
    rgb_out = cv.cvtColor(out, cv.COLOR_BGR2RGB)
    print(f"  rgb_out shape={rgb_out.shape}, dtype={rgb_out.dtype}, min={np.min(rgb_out)}, max={np.max(rgb_out)}")

