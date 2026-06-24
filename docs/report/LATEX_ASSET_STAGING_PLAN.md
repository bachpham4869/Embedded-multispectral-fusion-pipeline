# A4 LaTeX Asset Staging Plan

Status: staging plan plus copied figure inventory. Figure assets selected for this A4 plan have been copied into existing chapter asset folders under `HK252-DATN-142/`; no report `.tex` source was modified.

## Target Roots

| Target root | Intended content |
| --- | --- |
| `HK252-DATN-142/figures/ch6_evaluation/ml_classifier/` | Existing ML figure folder; selected cluster-aware and sensor domain-shift figures were copied here. |
| `HK252-DATN-142/figures/ch6_evaluation/fusion/` | Existing fusion figure folder; selected strict paired generated-fusion grids were copied here. |
| `HK252-DATN-142/tables/ch6_evaluation/ml_classifier/` | Prepared compact ML tables |
| `HK252-DATN-142/tables/ch6_evaluation/fusion/` | Prepared compact fusion tables |
| `HK252-DATN-142/tables/ch7_limitations/` | Optional limitation tables if the later patch uses external table files |

## Copied Figure Assets

| Source | Target name | Action | Caption/label source |
| --- | --- | --- | --- |
| `docs/figures/ml/cluster_aware/confusion_random_forest_200_current_config.png` | `figures/ch6_evaluation/ml_classifier/confusion_random_forest_200_cluster_aware.png` | copied | `prepared_latex/captions_and_labels.md` |
| `docs/figures/ml/raw_sensor_confidence_hist.png` | `figures/ch6_evaluation/ml_classifier/raw_sensor_confidence_hist.png` | copied | `prepared_latex/captions_and_labels.md` |
| `docs/figures/ml/paired_nir_feature_pca.png` | `figures/ch6_evaluation/ml_classifier/paired_nir_feature_pca.png` | copied; inspect readability before use | `prepared_latex/captions_and_labels.md` |
| `docs/figures/fusion/strict_paired_fusion_comparison_grid.png` | `figures/ch6_evaluation/fusion/strict_paired_fusion_comparison_grid.png` | copied | `prepared_latex/captions_and_labels.md` |
| `docs/figures/fusion/strict_paired_failure_cases_grid.png` | `figures/ch6_evaluation/fusion/strict_paired_failure_cases_grid.png` | copied for limitation/appendix option | `prepared_latex/captions_and_labels.md` |

## Files To Convert To LaTeX Tables Later

| Prepared snippet | Suggested target | Notes |
| --- | --- | --- |
| `docs/report/prepared_latex/ml_offline_model_comparison_table.tex` | inline in `ch6-evaluation.tex` or table input | Main ML result table |
| `docs/report/prepared_latex/ml_model_decision_table.tex` | inline in `ch6-evaluation.tex` or appendix | Main or appendix depending page space |
| `docs/report/prepared_latex/ml_class_decision_table.tex` | appendix or Ch6 compact version | Wide table; use `footnotesize` |
| `docs/report/prepared_latex/fusion_result_summary_table.tex` | inline in `ch6-evaluation.tex` | Main fusion result table |
| `docs/report/prepared_latex/per_bucket_evidence_table.tex` | inline in `ch6-evaluation.tex` | Main evidence status table |
| `docs/report/prepared_latex/limitations_table.tex` | inline in `ch7-conclusion.tex` | Compact limitations table |

Table snippets remain in `docs/report/prepared_latex/` until the LaTeX patch phase decides whether to inline them or move them under `HK252-DATN-142/tables/`.

## Git Tracking Note

The copied figure files exist in `HK252-DATN-142/`, but the repository `.gitignore` currently ignores the whole `HK252-DATN-142/` tree for new files. If these copied assets must be committed later, stage them explicitly with `git add -f` after the LaTeX patch is approved.

## Reference Only, Do Not Copy

| Source | Reason |
| --- | --- |
| Raw videos under `data/` or capture folders | Too large and not report assets |
| Model binaries or in-memory benchmark artifacts | Not needed in LaTeX and may be large |
| Full raw CSV metric tables | Keep as evidence artifacts; summarize in LaTeX |
| `docs/figures/ml/agent_labeled_sensor_confusion_matrix.png` | Not a main result; optional diagnostic only |
| Legacy `fusion_benefit_by_class.png` and `alpha_sweep_curve.png` | Superseded by current strict paired generated-fusion evidence |

## Resize Or Compress

- Check `strict_paired_fusion_comparison_grid.png` and `strict_paired_failure_cases_grid.png` dimensions before copying.
- Prefer one grid in the main chapter and move failure examples to limitations/appendix.
- Do not compress source artifacts in place. If needed, create a smaller report copy in the target LaTeX figure folder during the patch phase.
