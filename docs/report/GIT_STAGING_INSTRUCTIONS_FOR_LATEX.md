# Git Staging Instructions for LaTeX

**Status:** READY
**Branch:** `refactor/ml-taxonomy-eval-plan`
**Report Tree:** `HK252-DATN-142/`

> [!WARNING]
> Because `HK252-DATN-142/` is excluded by the project's `.gitignore` file, standard `git status` and `git add .` will **not** show or stage these LaTeX changes. You **MUST** use force-add (`git add -f`) for the specific files listed below.

---

## 1. List of Patched LaTeX Document Files

To stage the patched LaTeX report files on your local drive, run the following command:

```bash
git add -f \
  HK252-DATN-142/thesis.tex \
  HK252-DATN-142/chapters/front/abstract.tex \
  HK252-DATN-142/chapters/main/ch3-requirements.tex \
  HK252-DATN-142/chapters/main/ch4-system-design.tex \
  HK252-DATN-142/chapters/main/ch5-implementation.tex \
  HK252-DATN-142/chapters/main/ch6-evaluation.tex \
  HK252-DATN-142/chapters/main/ch7-conclusion.tex \
  HK252-DATN-142/chapters/back/design-docs.tex \
  HK252-DATN-142/chapters/back/sweeps.tex
```

---

## 2. List of Patched / New Table Assets

To stage the newly created and integrated dataset distribution and evaluation tables, run:

```bash
git add -f \
  HK252-DATN-142/tables/ch6_evaluation/ml_classifier/ml_dataset_class_distribution_table.tex \
  HK252-DATN-142/tables/ch6_evaluation/ml_classifier/ml_dataset_source_distribution_table.tex \
  HK252-DATN-142/tables/ch6_evaluation/ml_classifier/ml_offline_model_comparison_table.tex \
  HK252-DATN-142/tables/ch6_evaluation/ml_classifier/ml_model_decision_table.tex \
  HK252-DATN-142/tables/ch6_evaluation/ml_classifier/ml_class_decision_table.tex \
  HK252-DATN-142/tables/ch6_evaluation/ml_classifier/ml_domain_shift_summary_table.tex \
  HK252-DATN-142/tables/ch6_evaluation/fusion/fusion_result_summary_table.tex \
  HK252-DATN-142/tables/ch6_evaluation/fusion/per_bucket_evidence_table.tex \
  HK252-DATN-142/tables/ch7_limitations/limitations_table.tex
```

---

## 3. List of Reused Figure Assets

To stage the newly copied or active figure assets in the LaTeX directories, run:

```bash
git add -f \
  HK252-DATN-142/figures/ch6_evaluation/ml_classifier/confusion_random_forest_200_cluster_aware.png \
  HK252-DATN-142/figures/ch6_evaluation/ml_classifier/raw_sensor_confidence_hist.png \
  HK252-DATN-142/figures/ch6_evaluation/ml_classifier/paired_nir_feature_pca.png \
  HK252-DATN-142/figures/ch6_evaluation/fusion/strict_paired_fusion_comparison_grid.png \
  HK252-DATN-142/figures/ch6_evaluation/fusion/strict_paired_failure_cases_grid.png
```

---

## 4. List of Normal (Non-Ignored) Markdown Reports

These documentation and audit files under `docs/` are **not** ignored and can be staged using standard commands, or directly:

```bash
git add \
  docs/report/DATA_DISTRIBUTION_REPORT_CHECK.md \
  docs/report/REVIEWER_COVERAGE_FINAL_CHECK.md \
  docs/report/GIT_STAGING_INSTRUCTIONS_FOR_LATEX.md \
  docs/report/LATEX_PATCH_EXECUTION_REPORT.md \
  docs/report/LATEX_IMPACT_CHECK_REPORT.md \
  docs/report/LATEX_BUILD_STATUS.md \
  docs/report/LATEX_FINAL_QA_REPORT.md
```

---

## 5. Critical Warnings

> [!CAUTION]
> **NEVER** run `git add -f HK252-DATN-142/` blindly. Doing so will force-add binary files, raw build logs, intermediate compile cache (such as `.aux`, `.log`, `.toc`, `.pdf` files), or unapproved video/image files that do not belong in the Git repository.
> Only stage the specific files listed in Sections 1, 2, and 3 above.

---

## 6. Verification Command

After executing the stage commands above, you can verify your staging status by running:

```bash
git status --short
```

You should see all patched `.tex` files, `.png` files, and `.md` reports listed as staged (`A` or `M`).
