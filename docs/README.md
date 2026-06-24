# docs/ — Tài liệu dự án

Thư mục này chứa kế hoạch **đang tham chiếu**, bằng chứng thesis, bảng đo và hình minh họa — **không** chứa hướng dẫn cài đặt chính (xem [`README.md`](../README.md) ở root).

Các plan dài / research cũ ở root [`../legacy/`](../legacy/) — xem [`../legacy/README.md`](../legacy/README.md).

## Mục lục nhanh

| Mục | Đường dẫn |
| --- | --- |
| **Evidence register (thesis)** | [`PIPELINE_EVIDENCE_REGISTER.md`](PIPELINE_EVIDENCE_REGISTER.md) — map pipeline, tham số, **Q1–Q3 + non-claims** |
| **Scope & limitations** | [`THESIS_SCOPE_LIMITATIONS.md`](THESIS_SCOPE_LIMITATIONS.md) |
| **Bảng đo (timing / IQA / ML)** | [`tables/README.md`](tables/README.md) |
| **Hình (IQA / ML / ablation)** | [`figures/README.md`](figures/README.md) |
| **RPi runbook** | [`RPi_RUNBOOK.md`](RPi_RUNBOOK.md) |
| **Edge Impulse (future)** | [`EDGE_IMPULSE_FUTURE_WORK.md`](EDGE_IMPULSE_FUTURE_WORK.md) |
| **Manifest / eval protocol** | [`eval/manifest_schema.md`](eval/manifest_schema.md) |
| **Research ENV v4** | [`research_report_20260415_env_policy_branching_optical_v4.md`](research_report_20260415_env_policy_branching_optical_v4.md) |
| **Dataset trạng thái** | [`DATASET_LOCAL_STATUS.md`](DATASET_LOCAL_STATUS.md) |
| **Changelog docs** | [`CHANGELOG.md`](CHANGELOG.md) |

## Archive trong repo (`legacy/`)

| Mục | Đường dẫn |
| --- | --- |
| Toàn bộ Markdown archive (plans, deploy, research, thesis) | [`legacy/md/`](legacy/md/README.md) |
| Toàn bộ Python archive (monolith, script cũ) | [`../legacy/py/`](../legacy/py/README.md) |

## Liên kết cũ (đổi đường dẫn)

- `docs/tables/*` phẳng → [`tables/timing/`](tables/timing/), [`tables/iqa/`](tables/iqa/), [`tables/ml/`](tables/ml/)
- `docs/THESIS_OPEN_QUESTIONS.md` → đã gộp vào [`PIPELINE_EVIDENCE_REGISTER.md`](PIPELINE_EVIDENCE_REGISTER.md) § *Thesis open questions*
- `docs/figures/iqa_ab_appendix.md` → [`figures/iqa/iqa_ab_appendix.md`](figures/iqa/iqa_ab_appendix.md)
- Reliability PNG nguồn → [`figures/ml/reliability/`](figures/ml/reliability/)
- Ablation reliability PNG → [`figures/ml/ablation/`](figures/ml/ablation/) (`.joblib` / `.json` vẫn trong `models/ablation/`)
