# THESIS EVAL INTEGRATION REFINEMENT REPORT

Date: 2026-06-06
Author: Thesis Integration Refinement Agent

This report summarizes the corrections and refinements made to the thesis integration files to enforce label/ref hygiene, soften overclaims, and align all latency/BU-TIV citations with their evidence boundaries.

## 1. Files Changed

| File Path | Change Type | Summary of Changes |
| :--- | :--- | :--- |
| `HK252-DATN-142/chapters/main/ch7-conclusion.tex` | MODIFY | Refinement of the latency caveat under Objective O3 in the summary of results (line 21). Clarified that stage/session timing is profiled but standalone model latency remains open. |
| `HK252-DATN-142/tables/ch6_evaluation/sweeps/butiv_mad_benchmark_table.tex` | MODIFY | Renamed the latency column from `Mean Latency` to `Workstation Latency` to avoid implication of target hardware measurements. |
| `EVAL_BENCHMARK_DEFENSE_GUIDE.md` | MODIFY | Softened bucket-dispatch overclaim (from "đảm bảo đầu ra luôn đạt chất lượng tốt" to "tăng khả năng định tuyến đúng bucket xử lý trong mô phỏng offline") and MAD workstation runtime claims. |
| `THESIS_EVAL_INTEGRATION_WORK_REPORT.md` | MODIFY | Removed duplicate `tab:tab:` and `fig:fig:` prefixes and refined model latency caveats in remaining risks. |

## 2. Exact Wording Softened

- **Bucket Dispatch Quality:**
  - *Old (Defense Guide, Section 7):* "...đảm bảo đầu ra hình ảnh hiển thị cho người dùng luôn đạt chất lượng tốt."
  - *New (Defense Guide, Section 7):* "...tăng khả năng định tuyến đúng bucket xử lý trong mô phỏng offline."
- **MAD Runtime Location:**
  - *Old (Defense Guide, Section 8):* "...chứng minh thuật toán MAD siêu nhẹ và hoàn toàn đáp ứng khả năng chạy ngầm thời gian thực."
  - *New (Defense Guide, Section 8):* "...gợi ý tiềm năng tích hợp gọn nhẹ (lightweight integration potential) và hiệu năng chạy nền tốt."
  - *Old (Defense Guide, Section 8 Claims):* "Thuật toán hoạt động cực kỳ nhẹ nhàng (dưới 0.2 ms/khung hình), phù hợp làm bộ chỉ thị cảnh báo sớm chạy ngầm trên Raspberry Pi 4B."
  - *New (Defense Guide, Section 8 Claims):* "Thuật toán hoạt động cực kỳ nhẹ nhàng (0.1527 ms/khung hình trên workstation), gợi ý tiềm năng tích hợp chạy nền gọn nhẹ."
  - *Old (Defense Guide, Section 12 Checklist):* "...hoạt động chạy ngầm dưới 0.2 ms/khung hình."
  - *New (Defense Guide, Section 12 Checklist):* "...với tiềm năng tích hợp chạy nền gọn nhẹ dựa trên hiệu năng máy trạm (0.1527 ms/khung hình)."
  - *Old (BU-TIV Table):* Column header `Mean Latency`
  - *New (BU-TIV Table):* Column header `Workstation Latency`
- **Latency Caveat Wording:**
  - *Old (Conclusion, line 21):* "...claims remain caveated because user-confirmed sensor labels, captured runtime fusion triples, raw radiometric thermal arrays, and current RF100/RF200 target-hardware latency are not yet available."
  - *New (Conclusion, line 21):* "...claims remain caveated because user-confirmed sensor labels, captured runtime fusion triples, and raw radiometric thermal arrays are not yet available; stage/session timing has been profiled; standalone RF100/RF200 feature+predict latency artifact should be cited if available, otherwise remains separate from full mode-matrix acceptance."

## 3. Labels/Refs Fixed

- Duplicated prefixes `tab:tab:ch6-butiv-mad-benchmark` and `fig:fig:ch6-mad-anomaly` in `THESIS_EVAL_INTEGRATION_WORK_REPORT.md` were corrected to `tab:ch6-butiv-mad-benchmark` and `fig:ch6-mad-anomaly`.
- All newly added tables under `HK252-DATN-142/tables/ch6_evaluation/` were verified to have unique labels (`tab:ch6-ml-manual-label-eval`, `tab:ch6-domain-shift-features`, `tab:ch6-domain-shift-mitigation`, `tab:ch6-butiv-mad-benchmark`).

## 4. Build Status

- **Status:** **TOOLCHAIN_BLOCKED**
- **Reason:** pdflatex, xelatex, latexmk, tectonic, and bibtex are not available on the workstation. No software was installed.

## 5. Remaining Risks

1. **Native MI48 Anomaly Labels Pending:** MAD benchmark on BU-TIV acts as surrogate evidence; native MI48 sequences still need gaging and validation.
2. **Offline Log Simulation Gap:** The no-retrain fallback decision policy was evaluated using offline feature logs. Direct on-device pipeline checks are required to verify stability.
3. **Standalone RF100/RF200 Hardware Latency:** Standalone feature extraction and predict timings on the Raspberry Pi 4B target board remain open.
