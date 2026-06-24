# THESIS REFINEMENT WORK REPORT

Date: 2026-06-04

## 1. Executive summary

This pass refined the thesis manuscript and thesis-facing captions/tables to make the defense position more positive while keeping each claim inside its evidence boundary. The main correction is now explicit throughout the manuscript: the 584 Q1 rows are treated as live-captured paired NIR/thermal-display input rows, while the quantitative fusion comparison remains an offline-generated fusion evaluation from those captured pairs.

No source code, raw datasets, model artifacts, session JSON, fusion captures, or benchmark outputs were intentionally changed in this pass. The current worktree still contains unrelated tracked diffs outside this thesis-refinement scope under `src/` and `tools/`; they were left untouched. The thesis tree is ignored by Git in this checkout, so changed thesis files were verified by direct path inspection rather than Git diff.

Static validation passed for the requested unsafe wording patterns. The TeX toolchain is unavailable on this machine, so PDF build verification is TOOLCHAIN_BLOCKED. A fast pytest subset passed with plugin autoload disabled: 6 passed, 4 skipped.

## 2. Files changed table

| File | Change type | Summary |
|---|---|---|
| `HK252-DATN-142/chapters/front/abstract.tex` | Manuscript wording | Reframed the abstract around captured paired inputs, offline-generated fusion, Q2 offline IQA, RF200/RF100, partial target-hardware profiling, and pending validation gates. |
| `HK252-DATN-142/chapters/main/ch1-introduction.tex` | Manuscript wording | Replaced broad military/product claims with a binocular-style research prototype framing; made <=50 ms a design target unless full mode-matrix acceptance exists. |
| `HK252-DATN-142/chapters/main/ch2-background.tex` | Manuscript wording | Changed surrogate thermal sweep language from validation to support for EMA/Kalman parameter choices. |
| `HK252-DATN-142/chapters/main/ch3-requirements.tex` | Manuscript wording | Rewrote Q1/Q2/Q3 as evidence-bound research questions; updated the pytest inventory to 51 modules as of 2026-06-04; converted field validation phrasing to field profiling where appropriate. |
| `HK252-DATN-142/chapters/main/ch4-system-design.tex` | Manuscript wording | Removed exact stale module-count wording; made timing/fusion language evidence-bounded. |
| `HK252-DATN-142/chapters/main/ch5-implementation.tex` | Manuscript wording/caption | Reframed MAD as an anomaly indicator; clarified thermal figures as offline replay/surrogate support; clarified fusion as offline-generated from live-captured pairs. |
| `HK252-DATN-142/chapters/main/ch6-evaluation.tex` | Manuscript wording/captions | Reworked Q2/Q3/ML/timing/Q1 results to separate offline, surrogate, sensor-proxy, and partial target-hardware evidence. |
| `HK252-DATN-142/chapters/main/ch7-conclusion.tex` | Manuscript wording | Made conclusion positive but bounded: implemented prototype, reproducible evidence, offline ML, bucket IQA, paired-input fusion audit, with runtime/label/mode-matrix gates pending. |
| `HK252-DATN-142/chapters/back/test-suite.tex` | Appendix wording | Updated Appendix B test-suite wording to 51 pytest modules as of 2026-06-04. |
| `HK252-DATN-142/chapters/back/sweeps.tex` | Appendix wording/caption | Marked fusion alpha and thermal sweeps as proxy/surrogate evidence, not captured runtime fusion or live MI48 validation. |
| `HK252-DATN-142/chapters/back/design-docs.tex` | Appendix table wording | Changed simulation/sweep rows from validated to supported where the evidence is simulation or surrogate. |
| `HK252-DATN-142/tables/ch6_evaluation/fusion/fusion_result_summary_table.tex` | Thesis-facing table caption | Added `GENERATED_OFFLINE` evidence-class wording and live-captured paired input caveat. |
| `HK252-DATN-142/tables/ch6_evaluation/fusion/per_bucket_evidence_table.tex` | Thesis-facing table caption | Clarified forced offline evidence versus runtime bucket performance. |
| `HK252-DATN-142/tables/ch6_evaluation/ml_classifier/ml_domain_shift_summary_table.tex` | Thesis-facing table caption | Added `SENSOR_PROXY_UNLABELED` wording and excluded sensor accuracy claims. |
| `HK252-DATN-142/tables/ch7_limitations/limitations_table.tex` | Thesis-facing table wording | Added captured runtime fusion triples as a not-measured evidence gate. |
| `HK252-DATN-142/refs/example.bib` | Bibliography metadata | Corrected clearly wrong Dark Channel Prior metadata for `ref13` and `ref37`; no bibliography style or `\nocite{*}` change. |
| `THESIS_REFINEMENT_WORK_REPORT.md` | New report artifact | This work report. |

## 3. Claim wording changes table

| Area | Old risk | New wording direction |
|---|---|---|
| Q1 fusion | Runtime fusion could be read as fully validated. | "live-captured paired NIR/thermal inputs" plus "offline-generated fusion evaluation"; captured runtime fusion triples remain pending. |
| Q2 bucket dispatch | Offline still-image result could be read as live temporal validation. | "offline still-image cold-start evidence" and "real rain/transition sequence validation remains pending." |
| Q3 MAD | MAD could be read as a proven detector. | "MAD anomaly indicator" and "detector-accuracy validation deferred until labels exist." |
| ML classifier | RF metrics could be read as live-sensor accuracy. | "offline duplicate-cluster-aware benchmark"; sensor-domain runs are unlabeled proxy audits. |
| Latency/FPS | Stage timing could be read as full real-time acceptance. | "partial target-hardware profiling"; full mode-matrix acceptance remains pending. |
| Test suite | Stale pytest inventory count. | "As of 2026-06-04, 51 pytest modules." |
| Prototype scope | Military deployment/product language. | "binocular-style research prototype" and "defense-relevant observation" without fielded deployment claims. |

## 4. Positive evidence emphasized table

| Evidence | Path(s) checked | Manuscript use after refinement |
|---|---|---|
| 584 live-captured paired input rows | `artifacts/paired_eval/strict_paired_manifest.csv` = 584 rows | Supports paired-input capture/synchronization and offline-generated fusion comparison from captured pairs. |
| Q2 270 validation frames | `data/eval/nir_val/manifest_v2.csv` = 270 rows | Supports offline still-image bucket-dispatch/IQA analysis. |
| Q2 1,620 forced-bucket metric rows | `data/eval/iqa_runs/round_2026-04-28.csv` = 1,620 rows; `docs/thesis_eval/nir_enhancement/tables/iqa_round_2026-04-28.csv` = 1,620 rows | Supports per-bucket IQA comparison and fixed Bucket A versus adaptive dispatch discussion. |
| RF200/RF100 metrics | Existing ML thesis tables and docs under `docs/ml/`, `docs/tables/ml/`, `artifacts/ml/` | Supports offline duplicate-cluster-aware benchmark: RF200 accuracy 0.8263, balanced accuracy 0.7463, macro-F1 0.7362; RF100 remains lightweight candidate pending target latency. |
| Sensor-domain proxy audit | `artifacts/ml/sensor_domain_shift/raw_sensor_predictions.csv` = 590 rows; `manual_label_template_autofilled.csv` = 120 rows | Supports sensor-domain unlabeled audit and manual-label readiness, not sensor accuracy. |
| Timing/session evidence | `docs/thesis_eval/timing_performance/tables/session_index.csv` = 30 sessions | Supports partial target-hardware profiling and session telemetry context. |
| Long session evidence | Existing Chapter 3/7 inventory wording | Emphasizes two sessions over 5 minutes, including a 28.2-minute continuous session, without full mode-matrix acceptance. |
| Test inventory | `find tests -maxdepth 1 -name 'test_*.py'` = 51 | Supports updated Appendix B and Chapter 3/6 reliability wording. |

## 5. Weak evidence handled honestly table

| Weak area | How the manuscript now handles it |
|---|---|
| Missing strict runtime fusion triples | Q1 wording explicitly says fusion outputs are offline-generated and captured runtime fusion triples remain a future validation gate. |
| Missing raw/radiometric MI48 arrays | Q1/Q6 caveats say thermal-display inputs are available, but raw radiometric thermal validation is not. |
| Missing Q3 ground-truth labels | Q3 is an anomaly indicator smoke test only; precision, recall, F1, and false positives per minute are not claimed. |
| Pending sensor gold labels | ML sensor-domain results are described as unlabeled proxy/domain-shift audits, not live-sensor accuracy. |
| Pending RF RPi4 latency | RF100/RF200 feature-plus-predict latency is explicitly kept as a target-hardware gate. |
| Pending full mode matrix | Timing is partial target-hardware profiling; full mode-matrix acceptance remains pending. |
| Pending real rain/transition validation | Q2 results are offline still-image evidence; Bucket E/rain and transition/hysteresis require live temporal sequences. |
| Surrogate thermal sweeps | Parameter claims now say supported/characterized by simulation or surrogate sweep, not validated on live MI48 sequences. |

## 6. Citation/bibliography actions

| Item | Action |
|---|---|
| `\bibliographystyle{plain}` and `\nocite{*}` | Left unchanged. |
| Dark Channel Prior CVPR entry `ref13` | Corrected year/DOI to 2009 and `10.1109/CVPR.2009.5206515`. |
| Dark Channel Prior TPAMI entry `ref37` | Corrected DOI to `10.1109/TPAMI.2010.168`. |
| DCP duplicate key `ref36` | Left in place to avoid disrupting LINK-style numbering; it duplicates the CVPR 2009 metadata and should be reviewed after bibliography policy is confirmed. |
| MI48 support `ref19` | NEEDS_SOURCE: current entry is vendor-generic signal-chain support, not a strong MI48 module datasheet. |
| IMX290 support `ref22` | NEEDS_SOURCE: current entry cites Picamera2 documentation, which supports camera API usage but is weak for Sony IMX290 sensor spectral/hardware claims. |
| Online-doc access notes | Left unchanged unless clearly wrong; no invented sources were added. |

## 7. Figure/table actions

| Figure/table area | Action |
|---|---|
| Q1 fusion figures/tables | Captions now encode offline-generated fusion from live-captured paired inputs and explicitly exclude captured runtime fusion triples. |
| Q2 bucket/rain/transition figures/tables | Captions and prose now encode still-image cold-start/offline evidence and avoid live temporal validation claims. |
| Q3 MAD figures | Caption now calls it a smoke test/indicator output without detector-accuracy evidence. |
| ML figures/tables | Domain-shift table now uses `SENSOR_PROXY_UNLABELED`; RF comparison remains offline duplicate-cluster-aware. |
| Timing tables | Stage timing caption now says partial target-hardware profiling and not full mode-matrix acceptance. |
| Appendix parameter tables | Simulation/surrogate rows now use supported language. |
| Figure regeneration | No figures were regenerated in this pass. Recommended next action: rebuild low-resolution or screenshot-like figures from scripts into new output paths first, then inspect before replacing thesis figures. |

## 8. LaTeX/build status

| Check | Result |
|---|---|
| `latexmk` | NOT_FOUND |
| `pdflatex` | NOT_FOUND |
| `xelatex` | NOT_FOUND |
| `bibtex` | NOT_FOUND |
| `tectonic` | NOT_FOUND |
| Build status | TOOLCHAIN_BLOCKED |
| Pytest subset | `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/test_thesis_manifest_integrity.py tests/test_model_registry_integrity.py tests/test_session_inventory.py tests/test_fusion_evidence_readiness.py` -> 6 passed, 4 skipped in 2.75s. |
| Pytest caveat | Running pytest normally crashed before tests due `pytest-qt`/PyQt6: `Unable to embed qt.conf`; plugin autoload was disabled for the successful read-only subset. |
| Static unsafe wording scan | No matches for the requested high-risk phrases, stale counts, and unqualified runtime/sensor-real wording patterns. |
| Evidence-term scan | Confirmed thesis-facing use of `live-captured`, `offline-generated`, `captured runtime fusion triples`, `sensor-domain unlabeled`, `partial target-hardware profiling`, `full mode-matrix acceptance`, and 51 pytest modules. |

## 9. Remaining P0/P1 risks

| Priority | Risk | Current status | Required closure evidence |
|---|---|---|---|
| P0 | Captured runtime fusion triples missing | Q1 is bounded to offline-generated fusion from live-captured pairs. | Runtime NIR frame, thermal frame, fusion output, timestamp/homography sidecar, and per-frame timing captured from the active compositor. |
| P0 | Q3 has no ground-truth anomaly labels | MAD is only an anomaly indicator/smoke test. | 50-100 labeled thermal frames or sequences with frame-level/bounding-point labels and precision/recall/F1/false positives per minute. |
| P0 | Sensor-domain gold labels pending | ML sensor-domain audit is unlabeled/proxy. | User-confirmed sensor NIR labels and `tools/evaluate_sensor_labeled_predictions.py` result artifact. |
| P1 | RF100/RF200 target-hardware latency pending | Offline latency is not deployment latency. | RPi4 feature extraction plus predict benchmark with warmup, n, p50/p95/p99, CPU throttle, profile, and model hash. |
| P1 | Full mode-matrix acceptance pending | Timing is partial target-hardware profiling. | NIR-only, thermal-only, and fusion sessions across raw/throughput/quality profiles with FPS, drops, thermal reuse, and p95/p99 timing. |
| P1 | Real rain and transition validation pending | Bucket E/F remain offline/pseudo-sequence or still-image caveated. | Real temporal rain and dawn/dusk/transition sequences with dispatch stability and IQA/temporal metrics. |
| P1 | MI48/IMX290 citations weak | NEEDS_SOURCE recorded. | Vendor datasheets or official module documentation for MI48 radiometric/signal claims and IMX290 spectral/sensor claims. |

## 10. Next recommended steps

1. Collect a strict runtime fusion triple session: NIR input, thermal input, compositor output, timestamps, homography metadata, active profile, and timing per frame.
2. Complete the smallest Q3 label set: 50-100 thermal frames with binary frame labels and optional centroid/bounding-circle marks, then compute precision, recall, F1, false positives per minute, temporal persistence, and centroid stability.
3. Convert the 120-row sensor-domain manual-label template into user-confirmed labels and run the existing sensor-labeled evaluation tool.
4. Run RF100/RF200 feature-plus-predict latency on the Raspberry Pi 4B using the documented protocol.
5. Capture the full mode-matrix timing sessions before making any full real-time acceptance claim.
6. Replace weak MI48/IMX290 references with official datasheets or record the affected claims as hardware-context assumptions.
7. Install or use a TeX environment to run the full LaTeX cycle and check bibliography warnings after these edits.
