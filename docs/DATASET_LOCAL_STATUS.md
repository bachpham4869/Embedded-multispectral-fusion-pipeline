# Trạng thái dataset cục bộ (SmartBinocular ML)

**Cập nhật:** 2026-04-15 — Thêm link Kaggle *Landscape image colorization* (tham khảo). Trước đó (2026-04-09): dữ liệu Kaggle nằm **chỉ** dưới `data/weather/`; `data/_staging/` đã **gỡ**.

**Quyết định:** Không dùng Image2Weather — đủ MWD + Weather-Time + Weather 11-class cho offline.

---

## Chuẩn bố trí (canonical)

| Dataset | `--input-dir` cho `offline_pipeline.py` | Nhãn & file metadata |
|---------|----------------------------------------|----------------------|
| **MWD** | `data/weather/mwd` | Ảnh phẳng trong `dataset2/dataset2/*.jpg`. **Class = prefix tên file** (`cloudy`, `rain`, `shine`, `sunrise`) — regex lấy chữ cái đầu trước số. Không có JSON. |
| **Weather-Time** | `data/weather/weather_time` | `train_dataset/train.json`: mảng `annotations[]`, mỗi phần tử `{ "filename", "weather", "period" }`. Ảnh: `train_dataset/train_images/<name>.jpg`. **Nhãn ghép** `weather|period` (ví dụ `Cloudy|Morning`) → map trong `label_mapping.yaml` (`weather_time`). |
| **Weather 11-class** | `data/weather/weather11` | Thư mục lớp trong `dataset/<class>/` (tên lớp **lowercase**: `fogsmog`, `rain`, …). **Nhãn = tên thư mục.** Map → ENV trong `label_mapping.yaml` (`weather11`); nhiều lớp `env: null` (bỏ qua khi `--skip-null-labels`). |
| **Image2Weather** (optional) | `data/weather/image2weather` | `{Sunny,Cloudy,Rainy,Foggy,Snowy}/*.jpg` — **nhãn = tên thư mục.** Hiện không bắt buộc tải. |
| **Backlight** | `data/weather/backlight` | Ảnh phẳng `*.JPG` (hoặc định dạng trong `_IMG_EXTS`); **một lớp** → map `backlight` trong `label_mapping.yaml` (key `all`). |

`offline_pipeline.py` tự resolve: `mwd` → `.../dataset2/dataset2` nếu có; `weather11` → `.../dataset` nếu có. Có thể vẫn trỏ thẳng vào thư mục lá.

---

## Tóm tắt nhanh

| Dataset | Trạng thái | Ghi chú |
|---------|------------|---------|
| **MWD** (Kaggle) | Đã tải | ~1125 ảnh |
| **Weather-Time** (Kaggle) | Đã tải | Nhãn trong JSON + road images |
| **Weather 11-class** (Kaggle) | Đã tải | 6862 ảnh, 11 thư mục lớp |
| **Image2Weather** | Không dùng | Có thể bổ sung sau nếu cần |
| **Backlight** | Đã tải (local) | `python tools/offline_pipeline.py --dataset backlight --input-dir data/weather/backlight --output logs/ml/offline_backlight.jsonl` |
| **LLVIP / KAIST** | Optional | Thermal / night — xem [`legacy/md/OFFLINE_ML_PLAN.md`](../legacy/md/OFFLINE_ML_PLAN.md) |
| **Landscape image colorization** (Kaggle) | Tham khảo / optional | Ảnh cảnh quan (grayscale–color / colorization). **Nguồn:** [kaggle.com/datasets/theblackmamba31/landscape-image-colorization](https://www.kaggle.com/datasets/theblackmamba31/landscape-image-colorization). Chưa gắn `--dataset` trong `offline_pipeline.py` — tải về cục bộ nếu cần (gợi ý: `data/weather/landscape_colorization/`). |

---

## Chi tiết từng bộ

### `data/weather/mwd/`

- Ảnh: `dataset2/dataset2/*.jpg` (ví dụ `cloudy1.jpg`, `rain120.jpg`).
- Không có thư mục theo class; parser dùng **prefix tên file**.

### `data/weather/weather_time/`

- `train_dataset/train.json` — key top-level `"annotations"` (hoặc file là list).
- Mỗi annotation: `filename`, `weather` (`Sunny`, `Cloudy`, …), `period` (`Morning`, `Afternoon`, `Dawn`, `Dusk`, `Night`).
- Ảnh tương ứng: `train_dataset/train_images/`.
- `test_dataset/` tương tự nếu cần — pipeline mặc định đọc train.

### `data/weather/weather11/`

- `dataset/<tên lớp>/Ảnh` — 11 lớp (dew, fogsmog, hail, …).
- **Không** có file JSON nhãn riêng; nhãn = tên folder.

### `data/weather/image2weather/`

- Để trống nếu không dùng (tạo thư mục cục bộ khi cần).

---

## Dung lượng gần đúng (sau khi xóa zip trong `weather/`)

| Thư mục | Kích thước |
|---------|------------|
| `data/weather/mwd/` | ~95 MB |
| `data/weather/weather_time/` | ~474 MB |
| `data/weather/weather11/` | ~621 MB |

---

## Bước tiếp (ML)

1. Trích JSONL: `python tools/offline_pipeline.py --dataset <mwd|weather_time|weather11> --input-dir data/weather/<...> --output logs/ml/offline_<name>.jsonl`
2. `validate_schema.py` / `check_features.py` / `mix_datasets.py` → `data/training/optical_only.jsonl`
3. `python models/train_classifier.py --mode optical_only --dataset data/training/optical_only.jsonl`

Chi tiết lệnh: [`tools/README.md`](../tools/README.md).
