# Theo dõi kế hoạch: Indoor (tạm loại) + pipeline BGR + saturation

Tài liệu này liên kết **công việc trong plan** với **file/thư mục** cần đụng. Cập nhật cột trạng thái khi từng mục hoàn thành.

**Thứ tự thực hiện đã chốt:** làm **§4** (saturation) trước **§3** (baseline). Sau §4 đã xóa JSONL/`logs/ml` cũ và baseline cũ, rồi re-extract + train lại (§3) — hoàn tất.

| Trạng thái | Ý nghĩa |
| --- | --- |
| planned | Đã ghi trong plan, chưa merge code |
| done | Đã implement / docs đã đồng bộ |

---

## 1. Loại bỏ luồng BGR → gray; chỉ xử lý trên BGR

**Mục tiêu:** Không còn bước chuyển ảnh NIR sang grayscale như một nhánh xử lý chính: không `FrameCache.nir_gray`, không `COLOR_BGR2GRAY` trên các buffer pipeline; thống kê / motion / NIR enhancement dùng **BGR** hoặc phép lấy đại diện **không qua API gray** (ví dụ thống kê theo kênh, LAB `L`, hoặc biểu thức luminance vector hóa trên mảng nhỏ — chi tiết trong PR).

**File chính (rà soát và sửa):**

| File | Ghi chú |
| --- | --- |
| [`src/smartbinocular/utils.py`](../../src/smartbinocular/utils.py) | `FrameCache`: bỏ hoặc thay `nir_gray`; `build_frame_cache`, `nir_compute_gray_cached` |
| [`src/smartbinocular/main.py`](../../src/smartbinocular/main.py) | Cache glare/std/LK/JerkGate: bỏ phụ thuộc `nir_gray`, `nir_160` gray cho LK |
| [`src/smartbinocular/nir_pipeline.py`](../../src/smartbinocular/nir_pipeline.py) | `_nir_gray_*`, CLAHE/guided trên gray — chuyển sang đường BGR/LAB phù hợp |
| [`src/smartbinocular/motion.py`](../../src/smartbinocular/motion.py) | `_to_gray`, `nir_gray_small`, LK — input BGR hoặc 1 kênh lấy từ BGR không qua `BGR2GRAY` |
| [`src/smartbinocular/feature_extractor.py`](../../src/smartbinocular/feature_extractor.py) | Đặc trưng từ cache: không đọc `cache.nir_gray` nếu field bị xóa |
| [`src/smartbinocular/config.py`](../../src/smartbinocular/config.py) | Tên key `env_auto_nir_gray_std_*` có thể đổi nếu đo lường không còn “gray std” (optional, breaking nhẹ) |
| [`src/smartbinocular/env_presets.py`](../../src/smartbinocular/env_presets.py) | `nir_gray_std` — đổi semantics hoặc tên nếu metric mới |

**Không nằm trong scope gray-NIR:** [`display_pipeline.py`](../../src/smartbinocular/display_pipeline.py) dùng LAB cho grading (khác với `BGR2GRAY`).

**Trạng thái:** planned

---

## 2. Tạm loại indoor khỏi train offline

| File | Việc làm |
| --- | --- |
| [`tools/label_mapping.yaml`](../../tools/label_mapping.yaml) | Comment block `indoor_cvpr` + TODO |
| [`tools/offline_pipeline.py`](../../tools/offline_pipeline.py) | Disable / không đăng ký nguồn `indoor_cvpr` |
| Quy trình | Không merge `logs/ml/offline_indoor.jsonl` vào mix baseline |

**Trạng thái:** planned

---

## 3. Baseline retrain / eval

| File / artifact | Việc làm |
| --- | --- |
| JSONL đã merge (không indoor) | `mix_datasets` / split |
| [`models/train_classifier.py`](../../models/train_classifier.py) | Train + `--evaluate-model --test-dataset` |
| [`models/baseline/`](../../models/baseline/) | `*.joblib`, `*.json` metrics mới |

**Đã chạy (2026-04):** Xóa toàn bộ `logs/ml/*.jsonl`; re-extract 8 nguồn (`weather_time`, `mwd`, `weather11`, `darkface`, `exdark_street`, `glare_street`, `backlight`, `gray_nir`) → `mix_datasets` → `merged_logs_ml.jsonl` → `split_stratified_jsonl` → `from_logs_train.jsonl` / `from_logs_test.jsonl` → train `rf_from_logs_baseline.joblib` → eval trên test (accuracy ~0.84, balanced ~0.74 — xem `rf_from_logs_baseline.json`).

**Trạng thái:** done

---

## 4. Feature `nir_saturation_mean`

| File | Việc làm |
| --- | --- |
| [`src/smartbinocular/feature_schema.py`](../../src/smartbinocular/feature_schema.py) | `FEATURE_SET_CORE`, `FeatureRecord` |
| [`src/smartbinocular/feature_extractor.py`](../../src/smartbinocular/feature_extractor.py) | Tính mean kênh S từ BGR nhỏ (HSV) |
| [`tools/validate_schema.py`](../../tools/validate_schema.py) | Core bắt buộc |
| JSONL offline | Re-extract sau khi merge (bắt buộc — record cũ thiếu `nir_saturation_mean`) |
| [`src/smartbinocular/utils.py`](../../src/smartbinocular/utils.py) | `FrameCache.nir_small_bgr` — cùng resize với `nir_gray` |

**Trạng thái:** done (code); JSONL mới = bước §3

---

## 5. Profile RPi (~50 ms/frame)

Sau khi có saturation + sau khi refactor BGR: đo lại một vòng frame loop.

**Trạng thái:** planned

---

## 6. Đồng bộ README

Cập nhật thuật ngữ (pipeline BGR, indoor tạm loại, link tới trace này):

- [`README.md`](../../README.md) (root) — bảng chỉ dẫn có thêm link tới trace
- [`docs/README.md`](../../docs/README.md) — mục lục trỏ tới `legacy/README.md` (archive)
- [`tools/README.md`](../../tools/README.md) — bảng dataset: ghi chú `indoor_cvpr` tạm loại khỏi mix baseline
- [`models/README.md`](../../models/README.md) — dòng link tới trace cho baseline retrain
- [`src/README.md`](../../src/README.md) — link ngắn tới trace

Sau khi merge code (§1–§5), rà lại một vòng README nếu API/path đổi tên.

**Trạng thái:** partial (link đã thêm; chờ code merge để hoàn tất)

---

## Tài liệu plan gốc (Cursor, local)

Bản plan chi tiết nằm trong thư mục plan của Cursor (không commit): ví dụ `indoor_exclude_+_mono_sat_bef08fca.plan.md`, `indoor_bgr_saturation_055f7d2a.plan.md`.
