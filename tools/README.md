# SmartBinocular — ML tools (offline)

**Tổng quan thư mục:** công cụ **offline** để trích `FeatureRecord` → JSONL, gộp/cân bằng, split train/test stratified, validate/check — không chạy trong vòng lặp 60 FPS trên RPi. Tổng quan sản phẩm và thiết bị: [`README.md`](../README.md). Báo cáo sprint gần nhất: [`../weekly_report/week_3.md`](../weekly_report/week_3.md).

Thư mục này chứa script **chuẩn bị dữ liệu và train** cho classifier ENV (offline trên Mac/Linux), không chạy trên RPi. Chi tiết layout dataset cục bộ: [`docs/DATASET_LOCAL_STATUS.md`](../docs/DATASET_LOCAL_STATUS.md). Kế hoạch ràng buộc (C1–C10, archive): [`legacy/md/OFFLINE_ML_PLAN.md`](../legacy/md/OFFLINE_ML_PLAN.md).

## Công cụ

| File | Vai trò |
|------|---------|
| [`offline_pipeline.py`](offline_pipeline.py) | Đọc ảnh từ dataset đã tải → ghi **JSONL** feature (`FeatureRecord`) |
| [`label_mapping.yaml`](label_mapping.yaml) | Map nhãn gốc dataset → `ENV_CLASSES` trong `feature_schema.py` |
| [`validate_schema.py`](validate_schema.py) | Kiểm tra JSONL khớp schema (core features, kênh, nhãn) |
| [`check_features.py`](check_features.py) | Phân phối lớp, phương sai feature, cảnh báo chất lượng |
| [`mix_datasets.py`](mix_datasets.py) | Gộp / lọc / cân bằng nhiều JSONL → một file train (ví dụ `data/training/optical_only.jsonl`) |
| [`split_stratified_jsonl.py`](split_stratified_jsonl.py) | Chia **train / test** theo từng lớp (mặc định 85/15) từ thư mục per-class hoặc một JSONL đã merge |
| [`threshold_sweep.py`](threshold_sweep.py) | Sweep **τ₁** (max proba) trên tập test; ghi [docs/tables/ml/threshold_sweep.csv](../docs/tables/ml/threshold_sweep.csv) |
| [`secondary_threshold_sweep.py`](secondary_threshold_sweep.py) | Sweep **τ₂** (top-2) với `compose_env_from_ml_top2` thật; ghi [secondary_threshold_sweep.csv](../docs/tables/ml/secondary_threshold_sweep.csv) |

**Lý do chọn cặp ngưỡng toàn cục (τ₁, τ₂), ý nghĩa sweep vs runtime:** [`../docs/tables/ml/ML_GATE_RATIONALE.md`](../docs/tables/ml/ML_GATE_RATIONALE.md).

Train model: [`../models/train_classifier.py`](../models/train_classifier.py) (không nằm trong `tools/`).

## Feature set version

**Current:** `FEATURE_SET_OPTICAL_ONLY` = **12 features** (`nir_blue_mean_ema` added as feature #12 in Task 5).

> **Breaking change:** JSONL files generated with the **old 11-feature schema** are incompatible
> with the current `models/train_classifier.py`. The trained bundle will be rejected by
> `EnvClassifier._load` (feature_set mismatch → `available=False`). The system degrades
> safely to rule-based routing until a 12-feature bundle is deployed.

**Retrain steps after schema bump:**
1. Regenerate JSONL: `python tools/offline_pipeline.py ...` — output will include `nir_blue_mean_ema`.
2. Split: `python tools/split_stratified_jsonl.py ...`
3. Train: `python models/train_classifier.py --mode optical_only --dataset data/training/...`
4. `rsync` new bundle to RPi; set `ML_MODEL_PATH` in config.

`nir_blue_mean_ema` (B-channel mean EMA of `nir_small_bgr`, α=0.55) is **distinct** from
`main.py`'s existing `nir_b_ema` (brightness scalar EMA). Both coexist; different fields,
different semantics. The naming-collision guard is enforced by
`tests/test_feature_schema.py::test_nir_b_ema_not_in_feature_set`.

## Dataset trên đĩa (`data/weather/`)

| `--dataset` | `--input-dir` (khuyến nghị) | Cách gắn nhãn |
|-------------|----------------------------|----------------|
| `mwd` | `data/weather/mwd` | Prefix tên file (`cloudy*`, `rain*`, …) trong `dataset2/dataset2/` |
| `weather_time` | `data/weather/weather_time` | `train_dataset/train.json` → `weather` + `period`; ảnh trong `train_dataset/train_images/` |
| `weather11` | `data/weather/weather11` | Một thư mục con `dataset/<lớp>/`; nhãn = tên thư mục (lowercase) |
| `image2weather` | `data/weather/image2weather` | `{Sunny,…}/*.jpg` — optional, có thể bỏ qua |
| `darkface` | `data/weather/darkface` | Ảnh trong `image/` → map `night_clear` (xem `label_mapping.yaml`) |
| `exdark_street` | `data/weather/ExDark` | Chỉ thư mục Boat, Bus, Car, Motorbike → `normal_night` |
| `glare_street` | `data/weather/glare` | Mọi ảnh trừ đường dẫn có segment `mask` → `glare` |
| `backlight` | `data/weather/backlight` | Ảnh phẳng trong thư mục (hoặc lồng nhẹ) → `backlight` (một lớp, key `all` trong `label_mapping.yaml`) |

Các dataset `darkface`, `exdark_street`, `glare_street`, `backlight` dùng **`--max-env-samples`** (mặc định **3000**), **`--sample-seed`** (thứ tự xáo ảnh, mặc định 42), và **`--cap-seed`**: mức subsample sau jitter mặc định **khác mỗi lần chạy** (entropy OS); truyền `--cap-seed INT` nếu muốn cố định cap (ví dụ `--cap-seed 42` giống hành vi cũ khi cap luôn cùng một số).

**Ghi thêm theo lớp (để chia train/test dễ):** thêm `--by-label-dir logs/ml/by_label` — mỗi bản ghi cũng được **append** vào `logs/ml/by_label/<nhãn>.jsonl`. Trước khi chạy lại toàn bộ dataset, xóa hoặc dùng thư mục trống để không trộn lần chạy cũ.

Chi tiết và kích thước: [`docs/DATASET_LOCAL_STATUS.md`](../docs/DATASET_LOCAL_STATUS.md).

## Luồng đề xuất (từ ảnh → model)

1. **Có dataset trên đĩa** — xem bảng trạng thái trong `docs/DATASET_LOCAL_STATUS.md`.
2. **Trích feature** — một lần cho mỗi nguồn:

   ```bash
   cd /path/to/smartBinocular

   # Ví dụ: Weather 11-class (đường dẫn input khớp máy bạn)
   python tools/offline_pipeline.py \
     --dataset weather11 \
     --input-dir data/weather/weather11 \
     --output logs/ml/offline_weather11.jsonl \
     --by-label-dir logs/ml/by_label

   # Ví dụ ENV bổ sung (cap có jitter, mặc định upper=3000):
   # python tools/offline_pipeline.py --dataset darkface --input-dir data/weather/darkface \
   #   --output logs/ml/offline_darkface.jsonl
   # python tools/offline_pipeline.py --dataset exdark_street --input-dir data/weather/ExDark \
   #   --output logs/ml/offline_exdark.jsonl
   # python tools/offline_pipeline.py --dataset glare_street --input-dir data/weather/glare \
   #   --output logs/ml/offline_glare.jsonl
   # python tools/offline_pipeline.py --dataset backlight --input-dir data/weather/backlight \
   #   --output logs/ml/offline_backlight.jsonl

   # Các giá trị --dataset: image2weather | weather_time | mwd | weather11 | llvip_nir |
   #   darkface | exdark_street | glare_street | backlight
   ```

3. **Validate** — trước khi merge hoặc train:

   ```bash
   python tools/validate_schema.py --input logs/ml/offline_weather11.jsonl
   ```

4. **Phân tích** (tùy chọn):

   ```bash
   python tools/check_features.py --input logs/ml/offline_weather11.jsonl
   ```

5. **Gộp train set** — ví dụ optical-only:

   ```bash
   python tools/mix_datasets.py \
     --input logs/ml/offline_*.jsonl \
     --require-label-source dataset_original,manual \
     --output data/training/optical_only.jsonl \
     --max-per-class 3000
   ```

   **Hoặc** sau khi đã có `logs/ml/by_label/*.jsonl`, chia train/test theo lớp (85/15):

   ```bash
   python tools/split_stratified_jsonl.py \
     --input-dir logs/ml/by_label \
     --train-out data/training/optical_only_train.jsonl \
     --test-out data/training/optical_only_test.jsonl \
     --train-ratio 0.85 --seed 42 --max-per-class 3000
   ```

6. **Train baseline**:

   ```bash
   python models/train_classifier.py --mode optical_only --dataset data/training/optical_only.jsonl
   ```

Chạy `python tools/<script>.py --help` để xem đủ tham số.

## Bước tiếp theo (dataset + tools)

1. **Trích JSONL** từ ba bộ Kaggle đã có (`mwd`, `weather_time`, `weather11`) — đường dẫn `input-dir` như bảng trên.
2. **`mix_datasets.py`** → `data/training/optical_only.jsonl`, rồi **`train_classifier.py`** (baseline).
3. **Không bắt buộc Image2Weather** nếu đã đủ mẫu; bổ sung sau nếu cần đa dạng weather.
4. **Lớp ENV thiếu** — chủ yếu **field RPi**; `tools/label_session.py` trong plan **chưa có** trong repo.
5. **Production** — chỉ sau Phase 3C trên RPi (C6 trong `legacy/md/OFFLINE_ML_PLAN.md`).

## Phụ thuộc

Cần cài package ở root repo (editable) để import `smartbinocular`:

```bash
pip install -e .
```

Script thêm `src/` vào `sys.path` nên vẫn có thể chạy được khi chưa install, miễn là deps (OpenCV, numpy, …) đã có.
