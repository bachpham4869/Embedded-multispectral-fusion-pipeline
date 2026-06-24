import cv2 as cv
import numpy as np
from pathlib import Path
import json

from smartbinocular.thermal_pipeline import KalmanThermalBackground

def compute_interframe_diff(frames):
    diffs = []
    for i in range(1, len(frames)):
        diff = np.mean(np.abs(frames[i].astype(np.float32) - frames[i-1].astype(np.float32)))
        diffs.append(float(diff))
    return diffs

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

    # Baseline: No filter (Raw frames)
    raw_diffs = compute_interframe_diff(frames)
    
    # Process with Kalman Background
    bg = KalmanThermalBackground()
    kalman_bg_frames = []
    for frame in frames:
        bg.update(frame)
        if bg.cold_frame is not None:
            kalman_bg_frames.append(bg.cold_frame.copy())
            
    kalman_diffs = compute_interframe_diff(kalman_bg_frames)
    
    # Calculate statistics
    raw_mean = np.mean(raw_diffs)
    raw_std = np.std(raw_diffs)
    kalman_mean = np.mean(kalman_diffs)
    kalman_std = np.std(kalman_diffs)
    
    improvement = ((raw_mean - kalman_mean) / raw_mean) * 100
    
    print("Kalman Stability (Flicker Test) Results")
    print("-" * 50)
    print(f"Raw Background Diff    : Mean = {raw_mean:.3f}, Std = {raw_std:.3f}")
    print(f"Kalman Background Diff : Mean = {kalman_mean:.3f}, Std = {kalman_std:.3f}")
    print(f"Stability Improvement  : {improvement:.1f}%")
    
    # Save results
    out_dir = Path("docs/thesis_eval/timing_performance/tables")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "kalman_stability_results.json"
    
    results = {
        "raw_mean_diff": raw_mean,
        "raw_std_diff": raw_std,
        "kalman_mean_diff": kalman_mean,
        "kalman_std_diff": kalman_std,
        "improvement_pct": improvement
    }
    
    with out_file.open("w") as f:
        json.dump(results, f, indent=2)
        
    print(f"Saved results to {out_file}")

if __name__ == "__main__":
    main()
