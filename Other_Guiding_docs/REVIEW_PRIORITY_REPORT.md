# SmartBinocular thesis review priority report

Review mode: academic readiness review with evidence-recovery framing. This report is review-only and does not edit manuscript/source LaTeX.

## 1. Executive verdict

Verdict: major revision before submission/defense. The thesis has substantial engineering artifacts and a useful evidence registry, but the high-risk claims must be narrowed to match the actual evidence classes.

Core readiness issue: several sections already contain careful caveats, but the abstract, research-question framing, and some evaluation language still invite a reviewer to read offline/proxy evidence as runtime validation. The safest path is to keep the positive contributions, but label them precisely: `GENERATED_OFFLINE`, weak `RUNTIME_CAPTURED`, partial `TARGET_HW_PROFILED`, `SENSOR_PROXY_UNLABELED`, and `SURROGATE_SWEEP`.

Top blockers:

- Q1 fusion: strict paired evidence is generated offline; runtime captures exist but the current Q1 audit found 0 strict quantitative pairs.
- Q2 dispatch: forced-bucket evidence is still-image/offline, while rain and transition need temporal sequences; dispatch consistency is low and must be treated as a caveat.
- Q3 MAD: no ground-truth labels were found; it cannot be called a validated detector.
- ML: RF200/RF100 metrics are offline duplicate-cluster-aware; sensor-real labels and target-hardware RF latency remain pending.
- Latency/FPS: current evidence supports partial target-hardware profiling in named sessions, not full mode-matrix real-time acceptance.

## 2. Priority-ranked fix list

| Priority | Location | Issue | Evidence observed | Required action |
|---|---|---|---|---|
| P0 | Abstract, especially `HK252-DATN-142/chapters/front/abstract.tex` lines 3-9 and 37-39 | "Real-time dual-sensor fusion pipeline" and "advances the state of the art" read stronger than the evidence. | Q1 strict fusion is generated offline; runtime audit has 0 strict quantitative pairs; timing is partial. | Reword as a low-cost prototype with offline/generated fusion evidence and partial runtime profiling. Avoid "state of the art" unless tied to a precise contribution and cited baseline. |
| P0 | Chapter 3 RQ3, Chapter 6 Q3, Chapter 7 conclusion | RQ3 asks whether MAD is a validated detector, but evidence is no-label smoke output. | `docs/thesis_eval/open_questions/tables/mad_anomaly_results.json` exists; no label file found. | Either preserve RQ3 as an explicit negative/null finding or reframe as "what indicator behavior is observed"; never report detector validation. |
| P0 | Q1 in Abstract, Chapter 3, Chapter 6, Chapter 7 | Fusion utility can be misread as runtime-fusion validation. | `artifacts/paired_eval/strict_paired_manifest.csv` has 584 rows but no captured `fusion_output_path`; `q1_results/q1_report.md` reports 0 strict quantitative pairs. | Every Q1 quantitative claim needs "generated offline" or equivalent. Runtime captures should support feasibility/visual demonstration only. |
| P0 | Chapter 1 objective and Chapter 6 timing | The <=50 ms target, isolated microbench, stage timing, FPS, and full-loop acceptance can be conflated. | Timing JSONs and stage tables exist, but no full mode/profile/environment matrix and RF target latency is pending. | Split design target, profiled session timing, per-stage timing, ML latency, fusion compositor latency, and full acceptance. |
| P1 | Q2 abstract/Chapter 6 wording | "Confirmed" or "validated necessity" is too strong for still-image and surrogate evidence. | 270 NIR still frames, 1,620 forced-bucket rows, 12.6% dispatch agreement, pseudo-sequence rain sweep. | Reword as offline forced-bucket proxy evidence and dispatch-risk analysis. Add sequence recovery plan for rain/transition. |
| P1 | Chapter 1 ML objective vs Chapter 6 ML result | Objective mentions macro F1 >=0.90 on night-class subsets, while frozen RF200 macro-F1 is 0.7362 overall. | `docs/ml/ML_EVIDENCE_READINESS.md` reports RF200 accuracy 0.8263, balanced accuracy 0.7463, macro-F1 0.7362. | Clarify whether the >=0.90 target applies to a different subset; otherwise treat it as target not met or revise objective wording. |
| P1 | Chapter 3 and Appendix B test inventory | Test counts conflict. | Chapter 3 text/comments mention 22 files; Appendix B says 31 pytest modules; repo currently has 51 `tests/test_*.py` modules. | Regenerate unit-test inventory or change manuscript to a date-stamped count with script path. |
| P1 | Chapter 1 vs Chapter 4 architecture/module count | Module count conflicts. | Chapter 1 says 14 modules; Chapter 4 says sixteen Python modules. | Reconcile the count or remove exact count from prose. |
| P1 | Figures in Chapter 1, Chapter 3, Chapter 5, Chapter 6 | Several embedded images are low-DPI or too small for A4 print. | PDF image inventory shows examples around 112-130 DPI; MAD thumbnails appear around 51 DPI. | Redraw diagrams as vector PDF/SVG and enlarge/split thermal thumbnails. |
| P1 | Bibliography | Some technical references are weak or suspicious. | `plain` style and `\nocite{*}` are used; MI48/IMX290 entries rely on generic docs; DCP metadata appears duplicated/suspicious. | Audit citations, replace weak hardware specs with datasheets/vendor docs, and use a style expected by the school if required. |
| P2 | Evidence registry | Evidence index may be stale relative to current generated assets and claims. | `THESIS_EVIDENCE_INDEX.md` and `docs/thesis_eval/MANIFEST.md` exist; no freeze check was run in this pass. | Run evidence freeze checklist and manifest/model integrity tests after approval. |
| P2 | Captions for surrogate/generated figures | Some captions may not encode evidence class strongly enough. | Fusion, MAD, and rain/transition assets include generated/offline/surrogate evidence. | Captions should say generated offline, runtime captured weak pair, sensor proxy unlabeled, or surrogate sweep where applicable. |

## 3. Claim-evidence ledger

| Thesis claim | Current support | Evidence class | Readiness | Required wording boundary |
|---|---|---|---|---|
| Low-cost dual-sensor prototype was implemented | Source code, hardware/system chapters, runtime captures, session logs. | `RUNTIME_CAPTURED` plus implementation evidence | Supported | Keep as prototype/system implementation, not production deployment. |
| Q1 fusion improves observation utility | 584 strict paired offline rows and generated fusion metrics. | `GENERATED_OFFLINE` | Partially supported | Say generated/offline/no-reference utility only. |
| Q1 live runtime fusion was quantitatively validated | Runtime fusion images exist, but strict runtime triples were not found. | Weak `RUNTIME_CAPTURED` | Not supported | Only claim runtime feasibility/visual demonstration. |
| Homography/alignment is reliable | Homography file exists; sidecar quality indicates large drift in at least one manifest. | Metadata-only | Risky | Require alignment audit or field calibration capture. |
| Q2 adaptive bucket dispatch is useful | Forced-bucket still-image IQA and dispatch tables exist. | Offline proxy | Partially supported | Say offline forced-bucket proxy; avoid live-sequence validation. |
| Bucket E/rain temporal behavior is validated | Rain sweep exists but appears pseudo-sequence. | `SURROGATE_SWEEP` | Not validated | Say surrogate sensitivity only until real rain sequence exists. |
| Bucket F/hysteresis transition stability is validated | Open-question transition/stability artifacts exist, but real sequence scope is unclear. | Proxy/surrogate unless sequence is confirmed | Risky | Require sequence-based stability metric. |
| Q3 MAD is a validated detector | MAD smoke output exists; no labels found. | Smoke/proxy | Not supported | Use "indicator behavior" only. |
| RF200/RF100 classifier offline accuracy | Frozen ML docs/tables report cluster-aware metrics. | Offline duplicate-cluster-aware | Supported with caveat | Keep as offline benchmark. |
| Sensor-real classifier accuracy | Sensor predictions/features exist, and a 24-row agent/manual-reviewed subset exists, but labels are not user-confirmed gold labels. | `SENSOR_PROXY_UNLABELED` plus preliminary agent-label warning | Not supported yet | Requires user-confirmed manual label validation and labeled prediction evaluation. |
| RF feature+predict deployment latency | RPi4 latency script/protocol exists. | Pending `TARGET_HW_PROFILED` | Not supported yet | Requires RPi4 run output. |
| Runtime timing/FPS on prototype | Session JSONs and timing tables exist. | Partial `TARGET_HW_PROFILED` | Partially supported | Say profiled context/session, not full acceptance. |
| Full real-time acceptance across all modes/profiles | Full matrix not found. | Missing | Not supported | Requires fresh full mode-matrix benchmark. |
| Unit-test coverage/inventory | Tests exist, but manuscript counts conflict. | Repo test inventory | Needs cleanup | Use one generated count and date. |

## 4. Research-question audit

Q1: Directionally defensible if rewritten as generated/offline fusion utility plus runtime feasibility. The strongest existing assets are `artifacts/paired_eval/*`, `docs/fusion/*`, and `q1_results/*`. The dangerous wording is any statement implying captured runtime fusion utility was quantitatively validated.

Q2: Defensible as an offline forced-bucket and dispatch-risk study. The useful assets are `data/eval/nir_val/manifest_v2.csv`, `data/eval/iqa_runs/round_2026-04-28.csv`, `docs/tables/iqa/*`, and `docs/thesis_eval/bucket_dispatch/*`. Rain and transition behavior should remain caveated unless real temporal sequences are captured and evaluated.

Q3: The honest result is a negative or deferred validation result. Existing MAD artifacts can support a smoke test and qualitative examples, but not detector accuracy. A small 50-100 frame manual-label benchmark is the fastest path to preliminary metrics.

## 5. Methodology/evaluation audit

Strengths:

- The thesis already separates several evidence boundaries in later chapters, especially generated fusion, unlabeled sensor ML audit, and Q3 limitations.
- The repo has unusually rich artifacts for a student prototype: session JSONs, manifests, metric CSVs, generated tables, ML docs, and scripts.
- Cluster-aware ML reporting is much stronger than a naive random-split benchmark.

Risks:

- No-reference IQA metrics cannot by themselves prove human task utility or detection accuracy.
- Still images do not fully represent live dispatch, especially for rain and transition/hysteresis.
- Generated offline fusion is useful, but a reviewer will ask why runtime fusion output is not synchronously captured in the strict paired manifest.
- Sensor-domain ML data is currently unlabeled, so it can reveal domain drift but cannot support sensor-real accuracy.
- Stage timing tables need a single source of truth and exact context: hardware, profile, mode, warmup, n frames, p50/p95/p99, drops, thermal reuse, throttle status.

Recommended evaluation additions are specified in `RECOVERY_TEST_EVAL_BENCHMARK_PLAN.md`; the highest-value R0/R1 items are Q1 strict/runtime audit separation, Q2 forced-bucket/dispatch consistency, ML sensor-label evaluation, Q3 manual label subset, and evidence freeze tests.

## 6. Citation audit

No empty `\cite{}` commands were found in the reviewed source scan.

High-priority bibliography issues:

- `HK252-DATN-142/thesis.tex` uses `\bibliographystyle{plain}` and `\nocite{*}`. This may be acceptable only if the school allows alphabetic/numeric `plain` ordering and uncited bibliography entries. If IEEE/order-of-citation is required, this should be changed after approval.
- The dark channel prior references appear duplicated and metadata should be checked. One DCP entry is suspicious because it lists CVPR 2011 metadata; the repo also has CVPR 2009 and TPAMI 2011-style entries. Do not cite an unverified venue/DOI combination.
- The MI48/Senxor bibliography entry appears to describe a generic processing-chain citation rather than a direct MI48 datasheet/spec source. Hardware claims should cite the actual sensor datasheet/vendor documentation.
- The IMX290/Picamera2 citation is useful for software integration but is weak as a sensor-spec citation. Use Sony or board-vendor technical documentation for sensor specs.
- Online documentation entries should include access dates where required by the thesis format.

## 7. Figure/table audit

Figures that should be rebuilt or exported at higher quality:

- Chapter 1 architecture/processing diagrams: embedded at roughly 125-130 DPI in the PDF. Prefer vector PDF/SVG for system diagrams.
- Chapter 3 methodology flowchart: tall raster figure embedded around 112 DPI. Redraw as vector or split for A4 readability.
- Chapter 5 bucket-profile strips: several 720x238 raster figures appear around 114 DPI. Consider larger export or split panels.
- Chapter 6 MAD anomaly examples: source thumbnails are 240x62 and appear around 51 DPI. These are too small for print inspection; rebuild as larger panels with scale/legend.
- Chapter 4 hardware connection diagram appears low resolution and file metadata suggests a JPEG payload despite `.png` extension; re-export cleanly.

Caption rule: if a figure uses generated offline, surrogate, proxy, or unlabeled sensor evidence, the caption must say so. Figure polish must not be used to strengthen validation language.

## 8. LaTeX build audit

Local build status: not verified. The current shell does not have `latexmk`, `pdflatex`, `xelatex`, `bibtex`, or `tectonic` on PATH, so warnings, citation resolution, and PDF/source freshness could not be checked in this pass.

Compiled PDF reviewed: `/Users/phongpham/Downloads/HK252-DATN-142_2252614_2252057.pdf`, 150 pages, A4, generated by pdfTeX according to metadata.

Intended local verification command when TeX is available:

```bash
cd /Users/phongpham/Downloads/smartBinocular/HK252-DATN-142
pdflatex thesis.tex
bibtex thesis
pdflatex thesis.tex
pdflatex thesis.tex
```

Build claims still pending:

- No LaTeX warning audit.
- No overfull/underfull box audit.
- No citation-resolution audit from `.log`/`.blg`.
- No verification that source exactly matches the reviewed PDF.

## 9. Proposed edit plan

Emergency edits after evidence-review approval:

- Abstract: remove or narrow "state of the art" and real-time validation wording; keep prototype contribution and evidence-bounded achievements.
- Chapter 3: align RQ wording with evidence classes; make Q3 an explicit negative/deferred validation result unless labels are generated.
- Chapter 6: split Q1 generated offline vs runtime captured; split Q2 still-image vs sequence; split ML offline vs sensor-real; split stage timing vs full acceptance.
- Chapter 7: keep limitations visible and ensure conclusions do not exceed the caveats already stated.
- Appendix B and Chapter 3: regenerate test inventory count from current `tests/test_*.py`.
- Figures: rebuild low-DPI diagrams/thumbnails and add evidence-class caveats to captions.
- Bibliography: audit DCP, MI48, IMX290, online docs, and style requirements.

Recommended evidence order:

1. R0 dry-run/read-only checks and artifact registry audit.
2. R1 manual-label subsets for sensor ML and MAD.
3. R2 target-hardware runtime capture, RF latency, and full mode-matrix sessions.

## 10. Reviewer-style decision letter

The thesis presents a credible and artifact-rich SmartBinocular prototype, with a strong implementation base and an unusually transparent set of generated evidence files. The work is promising, but the manuscript is not yet ready to be defended as broadly validated. The main reason is not lack of effort; it is claim scope. Several findings are based on generated offline fusion, still-image IQA, unlabeled sensor-domain audits, or partial target-hardware timing. Those are useful results, but they must not be written as live runtime validation, sensor-real accuracy, or full real-time acceptance.

I would recommend major revision. The fastest path to readiness is to preserve the strong engineering contribution while narrowing the claims. If the recovery tests in `RECOVERY_TEST_EVAL_BENCHMARK_PLAN.md` are run, the thesis can add new evidence in the highest-risk areas without changing the underlying project scope. If the tests are not run before submission, the manuscript should still be acceptable only if it clearly frames Q1/Q2/Q3/ML/timing as prototype, offline, preliminary, or future-work evidence where appropriate.
