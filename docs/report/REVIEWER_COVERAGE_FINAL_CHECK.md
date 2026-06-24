# Reviewer Coverage Final Check

**Status:** PASS
**Date:** 2026-05-28
**Author:** Agent 4.6 (Final QA Phase)
**Target Document:** `HK252-DATN-142/thesis.tex` and Patched LaTeX Chapters

---

## 1. Executive Summary

This check verifies whether all 11 original reviewer concerns from the GVPB review are covered in the final LaTeX thesis report, where they are covered, and with what specific scientific caveats. All 11 concerns are successfully addressed in the text, ensuring a transparent, scientifically defensive, and highly rigorous document.

---

## 2. Reviewer Coverage Matrix

| Reviewer Concern | Coverage Status | Location in LaTeX Report | Key Scientific Caveat / Wording |
| :--- | :--- | :--- | :--- |
| **1. Bucket/class naming unclear** | `covered_strong` | Ch5 & Ch6 Sec 6.1, Table~\ref{tab:ch6-per-bucket-processing} | Terminology aligned with `night_hybrid_enhance`, `nir_mono_clahe`, `highlight_tone_map`, `fog_dehaze_lite`, `rain_temporal_median`, `dawn_dusk_blend`. |
| **2. Missing dataset distribution** | `covered_strong` | Ch6 Sec 6.2, Subsection "Dataset Distribution and Source Provenance", Tables \ref{tab:ch6-ml-class-distribution} and \ref{tab:ch6-ml-source-distribution} | Explicitly reports 14,094 reference rows, 11,981 train rows, and 2,113 test rows with full per-class/source counts and percentages. Caveats proxy-only modality. |
| **3. Imbalanced/low-support classes** | `covered_strong` | Ch6 Sec 6.2, Subsection "Dataset Distribution and Source Provenance" | Glare (2.84%), Backlight (2.61%), and Transition (3.66%) flagged as low-support. `transition` caveated as a transient runtime candidate, not a strong semantic class. |
| **4. 12 features too basic** | `covered_caveated` | Ch5 & Ch6 Sec 6.2, Subsection "Model Selection and Migration Gates" | `optical_12_baseline` remains the production baseline for lightweight CPU execution. Feature set v2 / 21 is treated strictly as a research candidate for future work. |
| **5. RF hyperparameters & parallelism** | `covered_strong` | Ch6 Sec 6.2, Subsection "Duplicate-Cluster-Aware Offline Benchmark" | RF200 and RF100 configurations fully detailed: $n\_estimators=200/100$, $depth=20$, $leaf=4$, $features="sqrt"$, balanced class weighting, and multi-core parallelism (\texttt{n\_jobs} = -1). |
| **6. Lack of model comparison** | `covered_strong` | Ch6 Sec 6.2, Table~\ref{tab:ch6-ml-cluster-aware-models} | Competitively compares RF200/RF100 against ExtraTrees, HistGradientBoosting, and MLP32. |
| **7. Lack of quantitative metrics** | `covered_strong` | Ch6 Sec 6.2 & 6.3, Tables \ref{tab:ch6-ml-cluster-aware-models}, \ref{tab:ch6-fusion-metrics}, and \ref{tab:ch6-per-bucket-processing} | Integrates ML metrics (Accuracy, Bal. Acc, Macro-F1, ECE, Brier) and image quality metrics (Foreground Contrast Gain, Edge Density, win-rates). |
| **8. Domain shift / real sensor deployment** | `covered_caveated` | Ch6 Sec 6.2, Subsection "Sensor Domain Shift Audit", Table~\ref{tab:ch6-ml-domain-shift} | Evaluates domain shift using 590 raw sensor frames and 584 paired NIR frames, caveated as proxy-scaling inference only, with user-confirmed labels listed as future work. |
| **9. Fusion too simple / lacking comparison** | `covered_strong` | Ch6 Sec 6.3, Subsection "Image Fusion and Multi-Modal Integration", Table~\ref{tab:ch6-fusion-metrics} | Compares `foreground_mask_overlay` against `alpha_blend_baseline`, caveated as generated offline, not captured runtime. |
| **10. Fusion quality metrics missing** | `covered_strong` | Ch6 Sec 6.3, Table~\ref{tab:ch6-fusion-metrics} | Uses Foreground Contrast Gain and Edge Density; includes `display_heatmap_like` caveat for thermal input video. |
| **11. Per-bucket processing details** | `covered_strong` | Ch6 Sec 6.1, Table~\ref{tab:ch6-per-bucket-processing} | Maps out all 6 processing buckets (A--F); flags rain temporal median and dawn/dusk blend as "not measured" due to lack of paired sensor evidence. |

---

## 3. Deep-Dive Details on Reviewer Items

### Item 1 — Naming Consistency
Every section of the document has been checked to ensure terminology is clean and exactly matches the code and logical pipelines. General terms like "enhanced night algorithm" have been replaced by the specific bucket names such as `night_hybrid_enhance` and `nir_mono_clahe`.

### Item 2 & 3 — Dataset Distribution, Imbalance, and Transition
With the addition of **Subsection "Dataset Distribution and Source Provenance"** in Chapter 6, the exact dataset statistics have been mapped out.
- Max-to-min class imbalance is calculated as $7.55$ (train) and $7.58$ (test).
- Low-support classes (`glare`, `backlight`, `transition`) are designated as "provisional" and flagged for future targeted labeling.
- The `transition` class is caveated as a transient runtime candidate class, explaining why its support is low and why it is not considered a stable semantic environment class.
- Modality provenance is heavily caveated: it represents offline visible-light optical RGB images acting as proxy evidence.

### Item 4 — 12 Features Baseline
Chapter 5 and 6 detail that the 12 features in `optical_12_baseline` (comprising color, edge, and contrast metrics) are chosen due to their low computational footprint on target-hardware CPUs. Feature set v2 and 21 (which includes texture and gradient features) are documented as future research candidates.

### Item 5 — Random Forest Hyperparameters
The hyperparameter configuration of the baseline model is fully detailed in Subsection 6.2:
- $n\_estimators = 200$ (RF200) or $100$ (RF100)
- $max\_depth = 20$
- $min\_samples\_leaf = 4$
- $max\_features = \text{"sqrt"}$
- `class_weight = "balanced"` (correcting for class imbalance during training)
- $\texttt{n\_jobs} = -1$ (multi-core parallelism)

### Item 8 — Domain Shift Caveats
The domain shift audit under Table~\ref{tab:ch6-ml-domain-shift} details prediction behaviors on 590 raw sensor frames and 584 paired NIR frames, but explicitly notes that:
- It represents proxy-scaling inference behavior under a domain shift.
- No ground truth or sensor-real accuracy is claimed because no user-confirmed gold labels are currently available.
- Visual inspection results from the 24 agent-labeled visual subset are explicitly segregated and marked as preliminary and biased, rather than main accuracy figures.

---

## 4. Conclusion

All 11 concerns have been fully addressed in the text, and their associated data structures are completely populated in the LaTeX tables. No unresolved reviewer blockers remain.

**Verdict:** **100% COVERED**
