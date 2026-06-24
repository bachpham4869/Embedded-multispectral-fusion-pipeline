# A4 LaTeX Structure Audit

Status: A4 audit/preparation. No LaTeX source was modified.

## Main Project Structure

| Item | Finding |
| --- | --- |
| Main file | `HK252-DATN-142/thesis.tex` |
| Document class | `hcmut-thesis.cls` loaded as `\documentclass[final]{hcmut-thesis}` |
| Bibliography | `\bibliographystyle{plain}` and `\bibliography{refs/example.bib}` |
| Loaded packages in main file | `pgfplots`, `amsmath`, `amssymb`, `textcomp`, `cleveref`, `multirow`, `ltablex`, `listings`, `xcolor`, `blindtext` |
| Build script | No `Makefile`, `latexmkrc`, or shell build script found under `HK252-DATN-142/`. |
| Figure root | `HK252-DATN-142/figures/` |
| Table root | `HK252-DATN-142/tables/` |
| Current report asset policy | Existing assets are already copied into chapter-specific folders; A4 will only propose future staging. |

## Included Files

| Order | File |
| ---: | --- |
| Front | `chapters/front/declaration.tex` |
| Front | `chapters/front/acknowledge.tex` |
| Front | `chapters/front/abstract.tex` |
| Front | `chapters/front/abbreviations.tex` |
| 1 | `chapters/main/ch1-introduction.tex` |
| 2 | `chapters/main/ch2-background.tex` |
| 3 | `chapters/main/ch3-requirements.tex` |
| 4 | `chapters/main/ch4-system-design.tex` |
| 5 | `chapters/main/ch5-implementation.tex` |
| 6 | `chapters/main/ch6-evaluation.tex` |
| 7 | `chapters/main/ch7-conclusion.tex` |
| Appendix A | `chapters/back/design-docs.tex` |
| Appendix B | `chapters/back/test-suite.tex` |
| Appendix C | `chapters/back/sweeps.tex` |
| Appendix D | `chapters/back/code-excerpts.tex` |

## Current Evaluation Sections

| File | Section | A4 relevance |
| --- | --- | --- |
| `ch6-evaluation.tex` | `Evaluation Methodology` | Add concise evidence-tier framing before ML/fusion subsections. |
| `ch6-evaluation.tex` | `Machine Learning Classifier Evaluation (Research Question Q1)` | Primary patch target for cluster-aware model comparison and domain-shift caveats. |
| `ch6-evaluation.tex` | `Feature Ablation and Offline Domain Shift` | Replace stale 9-vs-12 emphasis with current duplicate-cluster-aware benchmark, v2/21 research-only caveat, and sensor-domain proxy evidence. |
| `ch6-evaluation.tex` | `Fusion Optimization and Utility` | Primary patch target for paired data protocol, generated-fusion results, and runtime-fusion limitations. |
| `ch7-conclusion.tex` | `Limitations and Lessons Learned` | Update current "no domain-shift evaluation" style wording with measured proxy/domain-shift evidence and remaining gates. |
| `ch5-implementation.tex` | `Multi-Sensor Image Fusion` | Later patch should distinguish implementation method from generated offline evaluation. |
| `ch5-implementation.tex` | `Machine Learning Inference Pipeline` | Later patch should align current feature/model rationale with A1 evidence. |

## Stale Or Weak Sections

| Location | Issue | Recommended A4 action |
| --- | --- | --- |
| `ch6-evaluation.tex` ML aggregate text | Reports older CV/night-class numbers without current duplicate-cluster-aware split context. | Replace with current RF200/RF100/MLP model comparison and caveats. |
| `ch6-evaluation.tex` feature ablation text | Presents old 9-vs-12 framing as the main domain-shift story. | Keep only if needed as historical context; main result should use current model comparison and sensor proxy evidence. |
| `ch6-evaluation.tex` fusion section | Implies fusion utility and alpha choice more strongly than current generated-offline evidence supports. | Replace with strict paired input, generated-fusion comparison, and caveats. |
| `ch7-conclusion.tex` ML limitations | Says no domain-shift evaluation has been conducted. | Update to "user-confirmed sensor accuracy is not measured; proxy domain-shift evidence exists." |
| `ch7-conclusion.tex` fusion limitations | Existing proxy wording references synthesized thermal; current evidence uses real paired NIR video plus thermal display/heatmap-like video but generated fusion. | Update with current strict paired and thermal modality wording. |

## Recommended Insertion Points

- ML evidence: `ch6-evaluation.tex`, immediately under `\section{Machine Learning Classifier Evaluation (Research Question Q1)}` and before/within `Aggregate Performance and Calibration`.
- ML implementation rationale: `ch5-implementation.tex`, under `Machine Learning Inference Pipeline`.
- Fusion protocol and result: `ch6-evaluation.tex`, replacing `Fusion Optimization and Utility`.
- Fusion implementation caveat: `ch5-implementation.tex`, under `Multi-Sensor Image Fusion`.
- Limitations: `ch7-conclusion.tex`, `Evaluation Scope Limitations`, `Hardware and Deployment Limitations`, and `Machine Learning Limitations`.
- Reviewer-response mapping: no current dedicated reviewer response section exists in the LaTeX chapter list; use appendix only if a later phase adds one.
