"""eval_latency.py — Evaluate inference latency for pipeline stages.

This script benchmarks the computational latency (Performance Optimization)
of the various enhancement buckets and processing modules under the 
'throughput' and 'quality' profiles.
"""

import time
import sys
import csv
from pathlib import Path

import cv2 as cv
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from smartbinocular.nir_pipeline import (
    HybridNIREnhancer,
    RainTemporalMedian,
    nir_anti_glare_bgr,
    nir_dehaze_lite,
    nir_nir_night_clahe,
    nir_transition_blend,
)

# Dummy thermal processing setup
class DummyThermalFilter:
    def __init__(self):
        self.alpha = 0.65
        self.prev = None
    def process(self, frame):
        if self.prev is None:
            self.prev = frame.astype(np.float32)
            return frame
        out = self.alpha * frame.astype(np.float32) + (1.0 - self.alpha) * self.prev
        self.prev = out
        return out.astype(np.uint8)

def benchmark_function(func, *args, iterations=50, warmup=10):
    # Warmup
    for _ in range(warmup):
        func(*args)
    
    start = time.perf_counter()
    for _ in range(iterations):
        func(*args)
    end = time.perf_counter()
    
    avg_latency_ms = ((end - start) / iterations) * 1000.0
    return avg_latency_ms

def main():
    # Setup test frames
    bgr_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    thermal_frame = np.random.randint(0, 255, (62, 80), dtype=np.uint8)
    
    results = []
    
    # Bucket A
    enh_throughput = HybridNIREnhancer(patch_size=3, detail_strength=0.0)
    enh_quality = HybridNIREnhancer(patch_size=5, detail_strength=0.25)
    
    lat_a_thru = benchmark_function(enh_throughput.process, bgr_frame)
    lat_a_qual = benchmark_function(enh_quality.process, bgr_frame)
    results.append({"stage": "Bucket A (Night)", "throughput_ms": lat_a_thru, "quality_ms": lat_a_qual})
    
    # Bucket B
    lat_b = benchmark_function(nir_nir_night_clahe, bgr_frame)
    results.append({"stage": "Bucket B (CLAHE)", "throughput_ms": lat_b, "quality_ms": lat_b})
    
    # Bucket C
    lat_c = benchmark_function(nir_anti_glare_bgr, bgr_frame)
    results.append({"stage": "Bucket C (Anti-Glare)", "throughput_ms": lat_c, "quality_ms": lat_c})
    
    # Bucket D
    lat_d = benchmark_function(nir_dehaze_lite, bgr_frame)
    results.append({"stage": "Bucket D (Dehaze)", "throughput_ms": lat_d, "quality_ms": lat_d})
    
    # Bucket E
    rain = RainTemporalMedian()
    lat_e = benchmark_function(rain.process, bgr_frame)
    results.append({"stage": "Bucket E (Rain)", "throughput_ms": lat_e, "quality_ms": lat_e})
    
    # Thermal Temporal
    tf = DummyThermalFilter()
    lat_tf = benchmark_function(tf.process, thermal_frame)
    results.append({"stage": "Thermal 3D-NR", "throughput_ms": lat_tf, "quality_ms": lat_tf})
    
    # Fusion Alpha
    def run_fusion():
        th_warped = cv.resize(thermal_frame, (640, 480))
        th_color = cv.cvtColor(th_warped, cv.COLOR_GRAY2BGR)
        return cv.addWeighted(bgr_frame, 0.55, th_color, 0.45, 0.0)
        
    lat_fusion = benchmark_function(run_fusion)
    results.append({"stage": "Alpha Fusion", "throughput_ms": lat_fusion, "quality_ms": lat_fusion})
    
    # Output to stdout
    print(f"{'Stage':<25} | {'Throughput (ms)':<15} | {'Quality (ms)':<15}")
    print("-" * 61)
    for r in results:
        print(f"{r['stage']:<25} | {r['throughput_ms']:<15.2f} | {r['quality_ms']:<15.2f}")
        
    # Write to CSV
    out_dir = ROOT / "docs/thesis_eval/timing_performance/tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "eval_latency_results.csv"
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["stage", "throughput_ms", "quality_ms"])
        writer.writeheader()
        writer.writerows(results)
    
    print(f"\nSaved results to {out_csv}")

if __name__ == "__main__":
    main()
