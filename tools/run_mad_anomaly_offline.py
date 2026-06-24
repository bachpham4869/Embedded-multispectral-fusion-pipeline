import cv2 as cv
import numpy as np
from pathlib import Path
import json

from smartbinocular.thermal_pipeline import KalmanThermalBackground, ThermalMADAnomalyDetector

def main():
    base_dir = Path("data/thermal/scaled_mi48_sequences/auto_clip")
    if not base_dir.exists():
        print(f"Error: {base_dir} not found.")
        return

    out_dir = Path("docs/thesis_eval/open_questions/figures/mad_anomaly_examples")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Initialize Kalman background model and MAD detector
    bg = KalmanThermalBackground()
    mad = ThermalMADAnomalyDetector(
        mad_z_thresh=3.0,
        min_fg_pixels=5,
        temporal_window=2,
        min_area=10,
        heat_weight=0.35
    )

    print("Running MAD anomaly detection offline evaluation...")
    print(f"Directory: {base_dir}")
    print("-" * 60)

    results = []

    # There are 120 frames in the sequence: seq_00000.png to seq_00119.png
    for i in range(120):
        img_name = f"seq_{i:05d}.png"
        img_path = base_dir / img_name
        
        if not img_path.exists():
            continue
            
        # Read as grayscale
        frame = cv.imread(str(img_path), cv.IMREAD_GRAYSCALE)
        if frame is None:
            continue
            
        # 1. Update background model
        bg.update(frame)
        
        # 2. Get heat map and foreground mask (lowered threshold to catch smaller anomalies)
        heat_map = bg.get_heat_map(frame)
        fg_mask = bg.get_foreground_mask(frame, threshold=5.0)
        
        # 3. Process with MAD anomaly detector
        score, is_active, blobs = mad.process(frame, fg_mask, heat_map)
        
        if is_active and len(blobs) > 0:
            res = {
                "frame": i,
                "file": img_name,
                "score": score,
                "is_active": is_active,
                "num_blobs": len(blobs),
                "blobs": blobs
            }
            results.append(res)
            print(f"Frame {i:03d} | Active: {str(is_active):<5} | Score: {score:.3f} | Blobs: {len(blobs)}")
            
            # Save visual example
            color_frame = cv.cvtColor(frame, cv.COLOR_GRAY2BGR)
            for b_idx, blob in enumerate(blobs):
                print(f"  -> Blob {b_idx}: Area={blob['area']:.1f}, Score={blob['score']:.3f}, Center=({blob['cx']:.1f}, {blob['cy']:.1f})")
                cx, cy = int(blob['cx']), int(blob['cy'])
                cv.circle(color_frame, (cx, cy), 15, (0, 0, 255), 2)
                cv.putText(color_frame, f"E1 ({blob['score']:.2f})", (cx+18, cy-5), cv.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
                
            # Create a combined visualization (frame, heatmap, fgmask)
            hm_color = cv.applyColorMap(heat_map, cv.COLORMAP_JET)
            fg_color = cv.cvtColor(fg_mask, cv.COLOR_GRAY2BGR)
            combined = np.hstack((color_frame, hm_color, fg_color))
            cv.imwrite(str(out_dir / f"anomaly_example_{i:03d}.png"), combined)

    # Output to a JSON file for analysis
    out_file = Path("docs/thesis_eval/open_questions/tables/mad_anomaly_results.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with out_file.open("w") as f:
        json.dump(results, f, indent=2)
        
    print("-" * 60)
    print(f"Evaluation complete. Found anomalies in {len(results)} frames.")
    print(f"Results saved to {out_file}")

if __name__ == "__main__":
    main()
