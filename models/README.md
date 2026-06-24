# models/ — Huấn luyện & artifact baseline

| File / thư mục | Mô tả |
| --- | --- |
| **`train_classifier.py`** | Train / CV / eval Random Forest ENV (`optical_only`, ablation, …) |
| **`baseline/`** | Sau khi train, artifact lớn (`.joblib`) thường nằm local / gitignored. Repo giữ **`rf_phase1_retrain_optical12.json`** (metrics/ECE cho `docs/tables/ml/ece_*.md`). Bundle mặc định cho sweep: `models/production/env_classifier.joblib`. |
| **`ei/`** | Edge Impulse FOMO person-in-dark INT8 model — place `person_in_dark_fomo_int8.tflite` here (gitignored; see `ei/README.md`). |
| **`production/`** | ENV classifier bundle sidecar (`env_classifier.json`); `.joblib` usually local. |

Luồng đầy đủ: [`tools/README.md`](../tools/README.md) → `split_stratified_jsonl` → `train_classifier.py`. **Sau khi train:** cập nhật sweep + ghi chú ngưỡng theo [`../docs/tables/ml/ML_GATE_RATIONALE.md`](../docs/tables/ml/ML_GATE_RATIONALE.md). **Đồ thị reliability ablation** (PNG) lưu tại [`../docs/figures/ml/ablation/`](../docs/figures/ml/ablation/) (`.joblib` / `.json` ablation trong `models/ablation/`). Khoa học / calibration backlog (tài liệu): [`../docs/research_report_20260421_thesis_improvement.md`](../docs/research_report_20260421_thesis_improvement.md), trích dẫn IEEE [`../LINK.md`](../LINK.md) §IX.

Lịch sử plan (indoor tạm loại, saturation, baseline): [`legacy/md/PLAN_INDOOR_BGR_SATURATION_TRACE.md`](../legacy/md/PLAN_INDOOR_BGR_SATURATION_TRACE.md).

## Feature set version

**Current:** `FEATURE_SET_OPTICAL_ONLY` = **12 features** (`nir_blue_mean_ema` added as feature #12 in Task 5).

> **Breaking change:** 11-feature bundles (legacy train runs) are rejected by
> `EnvClassifier._load` (feature_set mismatch → `available=False`). Use the 12-feature
> `FEATURE_SET_OPTICAL_ONLY` bundle (`models/production/env_classifier.*`).

**Retrain steps after Task 5 schema bump:**
1. `python tools/offline_pipeline.py ...` — regenerate JSONL with `nir_blue_mean_ema`.
2. `python models/train_classifier.py --mode optical_only --input data/training/...` — produces new bundle.
3. `rsync` new bundle to RPi; set `ML_MODEL_PATH` in config.

`train_classifier.py` asserts `len(feature_set) == 12` for `optical_only` mode and provides
an actionable error message if the JSONL was generated with the old 11-feature schema.
