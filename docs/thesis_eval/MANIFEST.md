# Thesis Evaluation — Artifact Manifest

Columns:
- **artifact_path**: path relative to repo root (or pre-existing location)
- **description**: one-line description of what the file contains
- **source_command**: command that generates or generated the file
- **thesis_section**: suggested chapter / section
- **host**: `Mac` | `RPi4` | `RPi4_PENDING`
- **status**: `available` | `THESIS_SKIP` | `RPi4_PENDING`
- **caption**: short note for thesis figure/table caption (checked by test_thesis_manifest_integrity)

Note: pre-existing artifacts under `docs/eval/` and `docs/tables/` are linked in place — they are NOT copied or moved.

---

## ML Classifier

| artifact_path | description | source_command | thesis_section | host | status | caption |
|---|---|---|---|---|---|---|
| `docs/tables/ml/threshold_sweep.csv` | F1 vs confidence threshold for three night classes | `python tools/threshold_sweep.py` | Classifier threshold selection | Mac | available | IQA proxy — night-class F1 on held-out optical features; domain shift to NIR not validated |
| `docs/tables/ml/secondary_threshold_sweep.csv` | Secondary top-2 hint rate vs τ₂ with fixed τ₁=0.62 | `python tools/secondary_threshold_sweep.py --primary 0.62` | Compositor secondary gate | Mac | available | IQA proxy — hint rate derived from optical test set |
| `docs/tables/ml/ece_night_classes.md` | ECE per night class (night_clear, normal_night, nir_night) | existing sidecar | Classifier calibration | Mac | available | ECE values from isotonic calibration; small N for nir_night |
| `docs/tables/ml/ece_all_env_classes.md` | ECE for all 9 environment classes | existing sidecar | Classifier calibration | Mac | available | ECE values; glare/transition/nir_night have small sample counts |
| `docs/thesis_eval/ml_classifier/figures/confusion_matrix.png` | Confusion matrix on held-out test set (2113 rows) at τ=0.62 | `python tools/render_ml_curves.py --out-dir docs/thesis_eval/ml_classifier/figures/` | Classifier accuracy | Mac | available | Trained on visible-spectrum data; NIR deployment is a domain shift — UNVERIFIED for field accuracy |
| `docs/thesis_eval/ml_classifier/figures/roc_per_night_class.png` | ROC curves for night_clear / normal_night / nir_night (OVR) | `python tools/render_ml_curves.py --out-dir docs/thesis_eval/ml_classifier/figures/` | Night-class discrimination | Mac | available | OVR ROC on optical test set; domain shift to NIR deployment — UNVERIFIED |
| `docs/thesis_eval/ml_classifier/figures/pr_per_night_class.png` | Precision-recall curves for three night classes | `python tools/render_ml_curves.py --out-dir docs/thesis_eval/ml_classifier/figures/` | Night-class precision-recall | Mac | available | OVR PR on optical test set; domain shift to NIR deployment — UNVERIFIED |
| `docs/thesis_eval/ml_classifier/figures/f1_tau_abstention.png` | F1 vs threshold with dual-axis abstention rate | `python tools/render_ml_curves.py --out-dir docs/thesis_eval/ml_classifier/figures/` | Threshold operating point | Mac | available | IQA proxy; "best τ" = best proxy on optical test set — not validated for task gain |
| `docs/thesis_eval/ml_classifier/figures/learning_curve.png` | Macro night-F1 vs training-sample fraction | `python tools/render_ml_curves.py --out-dir docs/thesis_eval/ml_classifier/figures/` | Sample efficiency | Mac | available | Small N for glare/transition/nir_night — confidence intervals wide for those classes |
| `docs/thesis_eval/ml_classifier/figures/feature_importance.png` | RF mean decrease impurity for 12 optical features | `python tools/render_ml_curves.py --out-dir docs/thesis_eval/ml_classifier/figures/` | RF feature attribution | Mac | available | MDI importances; nir_saturation_mean dominates — interpret with care |
| `docs/thesis_eval/ml_classifier/figures/reliability_grid.png` | 3×3 calibration reliability grid per env class | `python tools/compose_reliability_figure.py --stem rf_phase1_retrain_optical12 --layout 3x3 --out docs/thesis_eval/ml_classifier/figures/reliability_grid.png` | Classifier calibration | Mac | available | OOF CV reliability; small N for nir_night — bins may be sparse; domain shift to NIR UNVERIFIED |
| `docs/thesis_eval/ml_classifier/tables/per_class_summary.csv` | Per-class F1 / abstention / ECE summary table | `python tools/render_ml_curves.py --out-dir docs/thesis_eval/ml_classifier/figures/` | Classifier per-class performance | Mac | available | Optical test set; domain shift caveat applies |

---

## NIR Enhancement Quality

| artifact_path | description | source_command | thesis_section | host | status | caption |
|---|---|---|---|---|---|---|
| `docs/tables/iqa/clahe_clip_sweep.csv` | CLAHE clip value sweep IQA on bucket-B night frames | `python tools/sweep_clahe_clip.py` | CLAHE clip selection | Mac | available | IQA proxy (log_rms, pct_sat, pct_crushed) — not validated against subjective quality |
| `docs/tables/iqa/clahe_clip_sweep_summary.md` | Per-clip aggregated means for CLAHE sweep | `python tools/sweep_clahe_clip.py` | CLAHE clip selection | Mac | available | IQA proxy summary |
| `docs/tables/iqa/sweep_anti_glare_c.csv` | Anti-glare bucket grid IQA (high_pct × saturate_at × gamma) | `python tools/sweep_anti_glare.py --manifest data/eval/nir_val/manifest_v2.csv` | Anti-glare tuning | Mac | available | IQA proxy — not validated against subjective quality |
| `docs/tables/iqa/sweep_dehaze_d.csv` | Dehaze bucket omega sweep IQA | `python tools/sweep_dehaze_omega.py` | Dehaze tuning | Mac | available | IQA proxy — omega=0.85 is production default; higher values risk shadow crush |
| `docs/tables/iqa/sweep_hybrid_params.csv` | Hybrid bucket parameter sweep IQA | `python tools/sweep_hybrid_params.py` | Hybrid enhancer tuning | Mac | available | IQA proxy for HybridNIREnhancer parameter grid |
| `docs/tables/iqa/sweep_transition_blend.csv` | Transition blend sweep IQA | `python tools/sweep_transition_blend.py` | Transition bucket blend | Mac | available | IQA proxy for transition bucket blend parameter |
| `data/eval/iqa_runs/round_2026-04-28.csv` | Full NIR IQA round (1620 rows, all buckets, all env classes) | `python tools/run_nir_iqa_eval.py --note thesis_round` | NIR enhancement quality | Mac | available | IQA proxy (before/after per bucket); BRISQUE excluded — cold-start bias on short clips |
| `docs/thesis_eval/nir_enhancement/tables/iqa_round_2026-04-28.csv` | Symlink / copy of round_2026-04-28.csv for thesis tree | manual copy after IQA run | NIR enhancement quality | Mac | available | IQA proxy (before/after); not validated against subjective quality |
| `docs/thesis_eval/nir_enhancement/tables/rain_median_n_sweep.csv` | Rain temporal-median N sweep IQA (N∈{2,3,5,7}, 20 rain frames) | `python tools/sweep_rain_median_n.py` | Rain denoise tuning | Mac | available | IQA proxy — pseudo-sequence surrogates (noise jitter); latency non-monotonic at small N due to buffer dynamics |
| `docs/thesis_eval/nir_enhancement/tables/clahe_clip_sweep.csv` | Copy of CLAHE clip sweep for thesis tree | symlink from docs/tables/iqa/ | CLAHE clip selection | Mac | available | IQA proxy — not validated against subjective quality |

---

## Bucket Dispatch

| artifact_path | description | source_command | thesis_section | host | status | caption |
|---|---|---|---|---|---|---|
| `docs/tables/iqa/dispatch_consistency.md` | Dispatch consistency heatmap (manifest labels vs rule-inferred) | `python tools/check_dispatch_consistency.py --manifest data/eval/nir_val/manifest_v2.csv` | Dispatcher consistency | Mac | available | Rule-based env inference vs manifest ground-truth labels; mixed_edge is ambiguous |
| `docs/tables/iqa/mis_dispatch_matrix_summary.csv` | Mis-dispatch matrix summary | existing from check_dispatch_consistency.py | Dispatcher consistency | Mac | available | Same caveats as dispatch_consistency.md |
| `docs/thesis_eval/bucket_dispatch/tables/dispatch_consistency.md` | Dispatch consistency: rule vs manifest labels (270 frames, 12.6% agreement) | `python tools/check_dispatch_consistency.py --manifest data/eval/nir_val/manifest_v2.csv --out docs/thesis_eval/bucket_dispatch/tables/dispatch_consistency.md` | Dispatcher consistency | Mac | available | Rule-based offline eval; 12.6% agreement expected — offline misses live EMA; ML compositor covers this gap |
| `docs/thesis_eval/bucket_dispatch/tables/per_bucket_iqa.csv` | Per-bucket mean IQA aggregated from IQA round (6 buckets) | `python tools/build_open_questions.py` (inline derivation from round_2026-04-28.csv) | Per-bucket image quality | Mac | available | IQA proxy (mean log_rms, pct_sat per bucket) — not validated against subjective quality |
| `docs/thesis_eval/bucket_dispatch/figures/bucket_gallery.png` | Raw NIR → bucket A-F output grid (2 frames per env class) | `python tools/build_bucket_gallery.py --manifest data/eval/nir_val/manifest_v2.csv` | Bucket dispatcher illustration | Mac | available | Still-image cold-start; not live pipeline output |

---

## Fusion Utility

| artifact_path | description | source_command | thesis_section | host | status | caption |
|---|---|---|---|---|---|---|
| `docs/thesis_eval/fusion/tables/fusion_alpha_sweep_proxy.csv` | Fusion alpha sweep IQA (α∈{0.30…0.80}, 270 frames × 7 alphas) | `python tools/sweep_fusion_alpha.py --manifest data/eval/nir_val/manifest_v2.csv` | Fusion blending behavior | Mac | available | PROXY — dummy thermal uniform-gray; best alpha = min NIR contrast suppression; spot-check vs on-device pending |
| `docs/thesis_eval/fusion/tables/per_class_fusion_alpha.csv` | Per-env-class optimal fusion alpha from proxy sweep | `python tools/per_class_fusion_alpha.py --sweep docs/thesis_eval/fusion/tables/fusion_alpha_sweep_proxy.csv` | Per-environment alpha tuning | Mac | available | PROXY — best alpha = max IQA proxy; not validated against task performance |
| `docs/thesis_eval/fusion/tables/ab_fusion_vs_nir_proxy.csv` | A/B fusion vs NIR-only IQA comparison on manifest frames (270 rows) | `python tools/ab_fusion_vs_nir.py --manifest data/eval/nir_val/manifest_v2.csv` | Fusion vs NIR baseline | Mac | available | PROXY — dummy thermal; negative delta expected; real benefit requires live aligned captures |
| `docs/thesis_eval/fusion/figures/thermal_contribution_hist.png` | Histogram of pixel-level thermal contribution to fused output | `python tools/thermal_contribution_hist.py --manifest data/eval/nir_val/manifest_v2.csv --out-dir docs/thesis_eval/fusion/figures/` | Thermal contribution analysis | Mac | available | PROXY — dummy thermal; Δ threshold hand-chosen — UNVERIFIED; two Δ values for sensitivity |
| `docs/thesis_eval/fusion/figures/alpha_sweep_curve.png` | Line chart of IQA vs alpha stratified by env class | `python tools/per_class_fusion_alpha.py --sweep docs/thesis_eval/fusion/tables/fusion_alpha_sweep_proxy.csv` | Fusion blending behavior | Mac | available | PROXY — IQA proxy; best alpha not validated against task performance |

---

## Timing and Performance

| artifact_path | description | source_command | thesis_section | host | status | caption |
|---|---|---|---|---|---|---|
| `docs/tables/timing/stage_timing_summary.csv` | Per-stage timing statistics from RPi4 sessions | `python tools/measure_stage_timing.py --session-dir fusion_captures/metrics/` | Per-stage statistics | RPi4 | available | RPi4 sessions; short sessions (24-244 s); ≥5-min single-mode sessions pending |
| `docs/thesis_eval/timing_performance/tables/stage_timing_by_mode.csv` | Per-stage timing split by pipeline mode (imx/thermal/fusion) from 18 sessions | `python tools/plot_stage_timing.py --sessions fusion_captures/metrics/` | End-to-end performance | RPi4 | available | RPi4; 18 sessions (24-244 s each); short sessions — ≥5-min dedicated sessions pending |
| `docs/thesis_eval/timing_performance/figures/stage_timing_bars.png` | Grouped bar chart: per-stage latency by mode | `python tools/plot_stage_timing.py --sessions fusion_captures/metrics/` | End-to-end performance | RPi4 | available | RPi4; sum-of-stage-means per session; ≥5-min dedicated sessions pending |
| `docs/thesis_eval/timing_performance/figures/latency_hist_50ms.png` | Per-session total latency with 50ms reference line | `python tools/plot_stage_timing.py --sessions fusion_captures/metrics/` | Real-time budget | RPi4 | available | RPi4; per-session sums not per-frame; 50ms budget line shown for reference |
| `docs/thesis_eval/timing_performance/tables/transitions_per_minute.csv` | ENV/bucket transitions per minute from session JSONs (lower-bound estimate) | `python tools/aggregate_transitions.py --sessions fusion_captures/metrics/` | Mode-switch rate | RPi4 | available | RPi4; ESTIMATED lower bound — derived from distinct class counts; exact counts need per-frame logging |
| `docs/thesis_eval/timing_performance/tables/session_index.csv` | Index of all RPi4 sessions with mode/duration/host | manually populated | Device session index | RPi4 | RPi4_PENDING | RPi4; ≥5-min single-mode sessions needed for full session matrix |

---

## Thermal Pipeline

| artifact_path | description | source_command | thesis_section | host | status | caption |
|---|---|---|---|---|---|---|
| `docs/thesis_eval/thermal/tables/kalman_qr_sweep.csv` | Kalman Q/R grid sweep on thermal sequences | `python tools/sweep_kalman_qr.py --seq-dir data/eval/thermal_seq/` | Thermal tracker tuning | Mac | THESIS_SKIP | THESIS_SKIP: data/eval/thermal_seq/ is empty; requires recorded MI48 .npy sequences from RPi4 |
| `docs/thesis_eval/thermal/tables/3dnr_alpha_sweep.csv` | 3DNR temporal smoothing alpha grid sweep | `python tools/sweep_3dnr_alpha.py --seq-dir data/eval/thermal_seq/` | Thermal denoise tuning | Mac | THESIS_SKIP | THESIS_SKIP: data/eval/thermal_seq/ is empty; requires recorded MI48 .npy sequences from RPi4 |

---

## Reproducibility

| artifact_path | description | source_command | thesis_section | host | status | caption |
|---|---|---|---|---|---|---|
| `docs/thesis_eval/reproducibility/tables/iqa_host_delta.csv` | Cross-platform IQA delta (Mac vs RPi4) on identical offline inputs | `python tools/diff_iqa_runs.py round_<mac>.csv round_<rpi4>.csv` | Cross-platform parity | Mac+RPi4 | RPi4_PENDING | THESIS_SKIP until RPi4 IQA round available; acceptable tolerance ±0.05 log_rms |
| `docs/tables/ml/ML_GATE_RATIONALE.md` | Rationale for τ=0.62 and τ₂=0.20 threshold choices | existing from sweep analyses | Classifier threshold rationale | Mac | available | Gate rationale; based on optical test set |
| `models/production/env_classifier.json` | Production sidecar (ECE, feature importances, gate reference, CV metrics) | existing | Model integrity | Mac | available | Production sidecar; SHA-256 verified by test_model_registry_integrity |

---

## Open Questions

| artifact_path | description | source_command | thesis_section | host | status | caption |
|---|---|---|---|---|---|---|
| `docs/thesis_eval/open_questions/tables/fusion_benefit_by_class.csv` | Fusion vs NIR-only mean Δlog_rms grouped by env class | `python tools/build_open_questions.py` | Fusion benefit by environment class | Mac | available | PROXY — dummy thermal; negative delta expected; class N imbalance; real benefit requires live captures |
| `docs/thesis_eval/open_questions/figures/fusion_benefit_by_class.png` | Bar chart of mean Δlog_rms by env class | `python tools/build_open_questions.py` | Fusion benefit by environment class | Mac | available | PROXY — dummy thermal; all classes show similar suppression due to uniform gray thermal |
| `docs/thesis_eval/open_questions/tables/ml_stability.csv` | Transition rate comparison: raw argmax vs hysteresis simulation | `python tools/build_open_questions.py` | ML temporal stability | Mac | available | ESTIMATED from test JSONL label sequence; not live session replay; hysteresis N=3 |
| `docs/thesis_eval/open_questions/figures/ml_stability.png` | Bar chart of transitions/min: raw vs hysteresis | `python tools/build_open_questions.py` | ML temporal stability | Mac | available | ESTIMATED — simulated on test JSONL; live transitions may differ |
| `docs/thesis_eval/open_questions/tables/train_deploy_gap.csv` | Train-split macro_f1_night vs deployment proxy (optical vs NIR) | `python tools/build_open_questions.py` | Train/deploy gap | Mac | available | Optical test set F1 available; NIR deployment F1 is RPi4_PENDING — domain shift UNVERIFIED |
| `docs/thesis_eval/open_questions/tables/bucket_share.csv` | Bucket share distribution from RPi4 session frames_by_bucket | `python tools/build_open_questions.py` | Compositor rule coverage | RPi4 | available | RPi4 session data; lighting conditions may bias bucket distribution |
| `docs/thesis_eval/open_questions/figures/bucket_share.png` | Bar chart of per-bucket frame share | `python tools/build_open_questions.py` | Compositor rule coverage | RPi4 | available | Bucket F highlighted; hypothesis bucket F fires less than 5% of frames |
| `docs/thesis_eval/open_questions/tables/dcp_resolution_tradeoff.csv` | DCP dehaze IQA at two output resolutions (320×240 vs 160×120) | `python tools/build_open_questions.py` | Resolution/cost tradeoff | Mac | available | IQA proxy; fog frames only; generalization to other classes limited |
| `docs/thesis_eval/open_questions/figures/dcp_resolution_tradeoff.png` | Bar charts of log_rms and proc_ms at two DCP resolutions | `python tools/build_open_questions.py` | Resolution/cost tradeoff | Mac | available | IQA proxy; fog frames; not validated against subjective quality |
| `docs/thesis_eval/open_questions/tables/fps_consistency.csv` | FPS mean per session across 18 RPi4 sessions | `python tools/build_open_questions.py` | Multi-session consistency | RPi4 | available | RPi4; 18 short sessions (24-244 s); CV=24% — ≥5-min sessions needed to verify |
| `docs/thesis_eval/open_questions/figures/fps_consistency.png` | FPS per session with mean ±1σ reference lines | `python tools/build_open_questions.py` | Multi-session consistency | RPi4 | available | RPi4; short sessions — variance inflated; label short-session caveat in thesis |
| `docs/eval/ei_person_find_in_dark/baseline_train500_crop_t060/epoch_00_raw_t060_crop_area/summary.json` | Person detection baseline (centroid-hit F1=0.9701 at τ=0.60, crop) | `python tools/eval_ei_person.py ...` | Person detection evaluation | Mac | available | Offline FOMO; centroid_hit F1 = 0.9701; INT8 softmax cap limits τ≥0.80 |
| `docs/eval/sessions/2026-04-30/REPORT.md` | Session metrics report (18 sessions, per-stage timing tables) | existing from session analysis | Device session analysis | RPi4 | available | RPi4 sessions; short sessions — ≥5-min dedicated sessions pending |

---

## Notes

- All IQA proxy artifacts (log_rms, pct_sat, pct_crushed, hist_entropy) carry the caption note: "IQA proxy — not validated against subjective task performance."
- All PROXY artifacts (fusion A/B, alpha sweep with dummy thermal) require spot-check vs on-device capture before being treated as quantitative evidence.
- All RPi4_PENDING timing artifacts will be updated to `available` once session JSONs are added to `fusion_captures/metrics/`.
- THESIS_SKIP artifacts record the reason for non-availability; the thesis text must note data was not available at writing time.
- The `caption` column is scanned by `tests/test_thesis_manifest_integrity.py` for internal code patterns — do not use abbreviations of the form `X1`, `P1`, etc. in that column.
