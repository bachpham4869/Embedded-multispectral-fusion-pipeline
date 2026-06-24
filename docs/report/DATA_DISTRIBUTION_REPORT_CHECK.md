# Data Distribution Report Check

**Status:** PASS
**Date:** 2026-05-28
**Report Version:** Agent 4.6 (Final QA Phase)
**Target Chapter:** `HK252-DATN-142/chapters/main/ch6-evaluation.tex`
**Inserted Tables:**
- `tab:ch6-ml-class-distribution` (`HK252-DATN-142/tables/ch6_evaluation/ml_classifier/ml_dataset_class_distribution_table.tex`)
- `tab:ch6-ml-source-distribution` (`HK252-DATN-142/tables/ch6_evaluation/ml_classifier/ml_dataset_source_distribution_table.tex`)

---

## 1. Distribution Metrics Checklist

| Metric Required | Value in Evidence | Status in LaTeX Report | Location / Reference |
| :--- | :--- | :--- | :--- |
| **Total Reference Rows** | 14,094 | **Covered** | Ch6 Sec 6.2, Subsection "Dataset Distribution and Source Provenance" |
| **Train Rows** | 11,981 | **Covered** | Ch6 Sec 6.2, Table~\ref{tab:ch6-ml-class-distribution} |
| **Test Rows** | 2,113 | **Covered** | Ch6 Sec 6.2, Table~\ref{tab:ch6-ml-class-distribution} |
| **Split Policy** | Duplicate-cluster-aware split | **Covered** | Ch6 Sec 6.2, Subsection "Dataset Distribution and Source Provenance" |
| **Per-Class Support** | 9 ENV classes detailed | **Covered** | Table~\ref{tab:ch6-ml-class-distribution} (Train & Test counts and %) |
| **Source Provenance** | 8 source datasets detailed | **Covered** | Table~\ref{tab:ch6-ml-source-distribution} (Train & Test counts and %) |
| **Class Imbalance** | 7.55 (train), 7.58 (test) | **Covered** | Ch6 Sec 6.2, Subsection "Dataset Distribution and Source Provenance" |
| **Low-Support Classes** | Glare (2.84%), Backlight (2.60%), Transition (3.64%) | **Covered** | Ch6 Sec 6.2, Subsection "Dataset Distribution and Source Provenance" |
| **Transient Class Caveat** | `transition` marked as transient runtime candidate | **Covered** | Ch6 Sec 6.2, Subsection "Dataset Distribution and Source Provenance" |
| **Proxy Modality Caveat** | Optical RGB-proxy vs raw/paired sensor data | **Covered** | Ch6 Sec 6.2, Subsection "Dataset Distribution and Source Provenance" |

---

## 2. Detailed Per-Class Count & Support (Table~\ref{tab:ch6-ml-class-distribution})

The following exact numbers have been integrated into the LaTeX report and match the frozen offline proxy evidence:

- **night_clear:** Train: 2,364 ($19.73\%$) | Test: 417 ($19.73\%$)
- **normal_night:** Train: 1,995 ($16.65\%$) | Test: 352 ($16.66\%$)
- **normal_day:** Train: 2,077 ($17.34\%$) | Test: 366 ($17.32\%$)
- **fog:** Train: 1,311 ($10.94\%$) | Test: 231 ($10.93\%$)
- **rain:** Train: 1,097 ($9.16\%$) | Test: 194 ($9.18\%$)
- **glare:** Train: 340 ($2.84\%$) | Test: 60 ($2.84\%$)
- **backlight:** Train: 313 ($2.61\%$) | Test: 55 ($2.60\%$)
- **transition:** Train: 438 ($3.66\%$) | Test: 77 ($3.64\%$)
- **nir_night:** Train: 2,046 ($17.08\%$) | Test: 361 ($17.08\%$)
- **Total:** Train: 11,981 ($100.00\%$) | Test: 2,113 ($100.00\%$)

---

## 3. Detailed Source Provenance (Table~\ref{tab:ch6-ml-source-distribution})

To trace data leakage and dataset origin, the exact source counts and percentages are detailed:

- `offline_backlight`: Train: 313 ($2.61\%$) | Test: 55 ($2.60\%$) | Total: 368 ($2.61\%$)
- `offline_darkface`: Train: 2,364 ($19.73\%$) | Test: 417 ($19.73\%$) | Total: 2,781 ($19.73\%$)
- `offline_exdark_street`: Train: 1,995 ($16.65\%$) | Test: 352 ($16.66\%$) | Total: 2,347 ($16.65\%$)
- `offline_glare_street`: Train: 340 ($2.84\%$) | Test: 60 ($2.84\%$) | Total: 400 ($2.84\%$)
- `offline_gray_nir`: Train: 2,046 ($17.08\%$) | Test: 361 ($17.08\%$) | Total: 2,407 ($17.08\%$)
- `offline_mwd`: Train: 946 ($7.90\%$) | Test: 177 ($8.38\%$) | Total: 1,123 ($7.97\%$)
- `offline_weather11`: Train: 1,768 ($14.76\%$) | Test: 300 ($14.20\%$) | Total: 2,068 ($14.67\%$)
- `offline_weather_time`: Train: 2,209 ($18.44\%$) | Test: 391 ($18.50\%$) | Total: 2,600 ($18.45\%$)
- **Total:** Train: 11,981 ($100.00\%$) | Test: 2,113 ($100.00\%$) | Total: 14,094 ($100.00\%$)

---

## 4. Key Caveat and Audit Verifications

1. **Class Imbalance & Low-Support:** Class support drops significantly below the $10\%$ average for `glare`, `backlight`, and `transition`. This is documented transparently in Ch6.
2. **The Transition Class Role:** Explicitly caveated as a "transient, short-lived runtime candidate" rather than a strong semantic environment.
3. **No Sensor Modality Confusion:** The report explicitly clarifies that this dataset represents **offline, visible-light optical RGB images acting as proxy evidence**. It clarifies that **no active NIR or LWIR active streams from the physical device are present, nor are they labeled by actual users**.
4. **Leakage Audit Status:** Fully documents that while duplicate-cluster group splitting successfully eliminates exact JSON/SHA/dHash cross-split pairs, it is **not source-held-out** (8 source names overlap), ensuring zero overstated claims of split independence.

## 5. Conclusion

With the addition of the new subsection and Tables~\ref{tab:ch6-ml-class-distribution} and \ref{tab:ch6-ml-source-distribution}, the dataset distribution is no longer described in generic terms. It contains comprehensive per-class and source-level distribution metrics.

**Verdict:** The dataset distribution check is **100% COMPLETE and PASS**. The thesis is now fully ready for reviewer check regarding dataset provenance.
