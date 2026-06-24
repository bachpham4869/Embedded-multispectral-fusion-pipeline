# legacy/

Tổng quan repo: [`README.md`](../README.md).

Hành vi và kiến trúc **hiện tại** của pipeline: [`README.md`](../README.md), [`PIPELINE_EVIDENCE_REGISTER.md`](../docs/PIPELINE_EVIDENCE_REGISTER.md), [`md/THESIS_IMPROVEMENT_PLAN.md`](md/THESIS_IMPROVEMENT_PLAN.md) *(archive)*, [`weekly_report/week_3.md`](../weekly_report/week_3.md), [`CLAUDE.md`](../CLAUDE.md), [`CHANGELOG.md`](../docs/CHANGELOG.md). Nội dung trong `legacy/` là **lưu trữ** — không dùng làm spec vận hành trừ khi đã đối chiếu với code.

**Cấu trúc:** toàn bộ Markdown archive nằm trong [`md/`](md/README.md); toàn bộ Python archive trong [`py/`](py/README.md). File này (`legacy/README.md`) là mục lục tổng.

---

## Archived Python (`py/`)

Monolith và script cũ: [`py/fusion_live_optimized.py`](py/fusion_live_optimized.py) và các file trong [`py/README.md`](py/README.md). **Không import**; thay thế bởi `src/smartbinocular/` (`main.py`, `config.py`, `hardware.py`, `thermal_pipeline.py`, `nir_pipeline.py`, `display_pipeline.py`, `motion.py`, `metrics.py`, `env_presets.py`, `utils.py`, `ml_inference.py`, …).

---

## Archived planning & research (`md/`)

Các tài liệu dưới đây từng nằm dưới `docs/` hoặc root `legacy/`; gom vào `legacy/md/` vì **lẫn trạng thái cũ** với kế hoạch dài hạn. Phần còn **khớp code hiện tại** nên tìm trong [`CLAUDE.md`](../CLAUDE.md) và [`docs/CHANGELOG.md`](../docs/CHANGELOG.md) trước. **Danh sách đầy đủ + mô tả ngắn:** [`md/README.md`](md/README.md).

| File | Phần đã lỗi thời / không còn khớp code | Phần vẫn đáng đọc lại / tham khảo |
| --- | --- | --- |
| [`md/ML_PER_CLASS_CONFIDENCE_PLAN.md`](md/ML_PER_CLASS_CONFIDENCE_PLAN.md) | §7 đã thực hiện (retrain 2026-04); ngưỡng và ECE nay trong [`docs/PIPELINE_EVIDENCE_REGISTER.md`](../docs/PIPELINE_EVIDENCE_REGISTER.md) và [`docs/tables/ml/ML_GATE_RATIONALE.md`](../docs/tables/ml/ML_GATE_RATIONALE.md). | Lịch sử hai scalar toàn cục; bảng ECE / sweep trong [`docs/tables/`](../docs/tables/README.md). |
| [`md/ML_HYBRID_SYSTEM_PLAN.md`](md/ML_HYBRID_SYSTEM_PLAN.md) | “Current state” đầu tài liệu; nhiều bảng so với pipeline hiện tại (bucket A–F, ML wired). | Ràng buộc ≤50 ms/frame, rule + ML, ý tưởng nghiên cứu. |
| [`md/OFFLINE_ML_PLAN.md`](md/OFFLINE_ML_PLAN.md) | Tên artifact / phase có thể khác `models/baseline/` hiện tại. | **C1–C10**, optical-first, schema JSONL. |
| [`md/PHASE2_SOFT_GATE.md`](md/PHASE2_SOFT_GATE.md) | Checklist D1–D7 có thể đã xử lý một phần. | Biên bản soft gate. |
| [`md/PLAN_INDOOR_BGR_SATURATION_TRACE.md`](md/PLAN_INDOOR_BGR_SATURATION_TRACE.md) | Cột trạng thái có thể lệch sau PR. | Lịch sử BGR / indoor / saturation. |
| [`md/research_report_20260415_env_classification_fusion_policy.md`](md/research_report_20260415_env_classification_fusion_policy.md) | Taxonomy sớm, chưa khớp `ENV_CLASSES`. | Khung ý tưởng ENV + fusion. |
| [`md/research_report_20260415_env_classification_fusion_policy_ultradeep.md`](md/research_report_20260415_env_classification_fusion_policy_ultradeep.md) | Rất dài; một phần taxonomy không khớp code. | Tham khảo bối cảnh. |
| [`md/research_report_20260415_env_policy_codebase_aligned_v3.md`](md/research_report_20260415_env_policy_codebase_aligned_v3.md) | Đã có bản v4 trong `docs/`. | Bước trung gian trước optical v4. |
| [`md/DEPLOY_ML_MODEL.md`](md/DEPLOY_ML_MODEL.md) | Checklist cũ. | Quy trình train → rsync, observe-only, τ gates. |
| [`md/DEPLOY_HARDENING.md`](md/DEPLOY_HARDENING.md) | N/A | `uv sync --frozen`, registry, **§8** luma gate. |
| [`md/RPi_FOLLOWUP_WORK.md`](md/RPi_FOLLOWUP_WORK.md) | Việc còn lại trên Pi có thể lệch. | Checklist thiết bị; số đo mới ở [`docs/tables/`](../docs/tables/README.md). |

**Research ENV/optical đang dùng (trong `docs/`):**

- [`docs/research_report_20260415_env_policy_branching_optical_v4.md`](../docs/research_report_20260415_env_policy_branching_optical_v4.md)

---

## Liên kết trong repo

Đường dẫn kiểu `docs/ML_HYBRID_SYSTEM_PLAN.md` hoặc `legacy/*.md` cũ đã chuyển thành **`legacy/md/…`**. Comment trong code trích **C1–C10** vẫn trỏ ngầm tới `legacy/md/OFFLINE_ML_PLAN.md`. Runbook RPi tích cực: [`docs/RPi_RUNBOOK.md`](../docs/RPi_RUNBOOK.md).
