import cv2 as cv
import numpy as np
from pathlib import Path
import json

from smartbinocular.thermal_pipeline import thermal_edge_enhance

def compute_metrics(frame):
    # Laplacian variance as a measure of sharpness
    laplacian = cv.Laplacian(frame, cv.CV_64F)
    variance = laplacian.var()
    
    # Noise measure: estimate noise standard deviation using a fast noise estimator (e.g., standard deviation of local variance)
    # A simple approach for thermal: we can just use std dev of the frame itself, or std of laplacian
    
    return float(variance)

def main():
    base_dir = Path("data/thermal/scaled_mi48_sequences/auto_clip")
    if not base_dir.exists():
        print(f"Error: {base_dir} not found.")
        return

    # Load all frames
    frames = []
    for i in range(120):
        img_path = base_dir / f"seq_{i:05d}.png"
        if img_path.exists():
            frame = cv.imread(str(img_path), cv.IMREAD_GRAYSCALE)
            if frame is not None:
                frames.append(frame)

    if not frames:
        print("No frames loaded.")
        return
        
    strengths = [0.0, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0]
    results = {}

    print("Edge Threshold Sweep Results")
    print("-" * 50)
    print(f"{'Strength':<10} | {'Laplacian Variance (Sharpness)':<30}")

    for strength in strengths:
        vars_all = []
        for frame in frames:
            enhanced = thermal_edge_enhance(frame, strength=strength)
            var = compute_metrics(enhanced)
            vars_all.append(var)
        
        mean_var = np.mean(vars_all)
        results[str(strength)] = mean_var
        
        print(f"{strength:<10.2f} | {mean_var:<30.2f}")
        
    # Save results
    out_dir = Path("docs/thesis_eval/timing_performance/tables")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "edge_sweep_results.json"
    
    with out_file.open("w") as f:
        json.dump(results, f, indent=2)
        
    print(f"Saved results to {out_file}")

if __name__ == "__main__":
    main()
