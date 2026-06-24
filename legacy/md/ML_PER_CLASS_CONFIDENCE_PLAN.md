# Plan: Phân tích ML theo từng lớp — chốt **một cặp ngưỡng toàn cục** (top-1 & top-2)

**Trạng thái:** §7 đã chạy (retrain 2026-04, `ece_by_class` + `threshold_sweep.csv` + figures); `config.py` ghi chú dựa trên sweep; `ml_confidence=0.62` / `ml_secondary=0.20` (xem `models/production/env_classifier.json`).  
**Giải thích tường tận (thesis, tiếng Việt):** [`../docs/tables/ml/ML_GATE_RATIONALE.md`](../../docs/tables/ml/ML_GATE_RATIONALE.md) — ý nghĩa τ₁/τ₂ trong `compose_env_from_ml_top2`, isotonic, macro-F1 đêm, P90(p₂), `hint_rate_of_ml`, hạn chế.  
**Mục tiêu:** chạy công cụ (reliability, ECE, sweep, v.v.) trên **đủ 9 `ENV_CLASSES`** để tính toán / đọc độ tin cậy **theo từng lớp**; từ đó **chọn một cặp ngưỡng toàn cục** `ml_confidence_threshold` (top-1) và `ml_secondary_confidence_threshold` (top-2) theo công thức hoặc quy tắc tổng hợp (mô tả trong luận văn). **Runtime không cần** `Dict` per-class — chỉ hai scalar.

**Quyết định thiết kế:** Ngưỡng dùng **mặc định toàn cục**; cùng ý nghĩa trên mọi tầng cấu hình (CONFIG, `RPI_THROUGHPUT_MAX_DEFAULTS`, …) vì đó là vài so sánh float, **không tăng độ nặng pipeline**. Không cần map lồng → **không** phát sinh bài toán shallow vs **deep merge**; nếu RPi cần số khác, ghi đè **cùng hai key float** (merge `update` bình thường).

---

## 1. Phạm vi lớp (phân tích, không phải 9 key CONFIG)

Căn cứ `ENV_CLASSES` trong `feature_schema.py` (9 chuỗi) — dùng cho đồ thị / bảng / sweep **theo lớp**:

`night_clear`, `normal_night`, `normal_day`, `fog`, `rain`, `glare`, `backlight`, `transition`, `nir_night`.

**Cấu hình chạy máy** chỉ cần **hai số** toàn cục cho hai gate; không bắt buộc 9 entry.

---

## 2. Ngữ nghĩa hai ngưỡng (giữ đúng `compose_env_from_ml_top2`)

| Ngưỡng | Trong `cfg` (toàn cục) | Ý nghĩa trong compositor (không đổi) |
|--------|------------------------|--------------------------------------|
| **Top-1 (primary)** | `ml_confidence_threshold` | Rule 1: nếu `proba_1` **dưới** ngưỡng → không dùng ML, fallback rule. |
| **Top-2 (secondary)** | `ml_secondary_confidence_threshold` | Rule 2: nếu `proba_2` **dưới** ngưỡng → bỏ gợi ý phụ, chỉ giữ top-1. |

*Phân tích theo lớp* (ECE, reliability, sweep) dùng để **lập luận** giá trị hai số trên, không tạo thêm cột cấu hình theo tên lớp.

**Lưu ý:** Khi `class_2` là `None`, nhánh top-2 không dùng. Khi `class_2` trùng `class_1`, compositor đã loại sớm.

---

## 3. Cấu hình (`config.py`) — hai scalar toàn cục

- `ml_confidence_threshold`  
- `ml_secondary_confidence_threshold`  

Cập nhật **sau** khi rút từ công thức / bảng (paper hoặc `docs/tables/…`).

**Preset RPi** có thể ghi đè **cùng hai key** nếu cần; không dùng dict lồng cho ngưỡng.

---

## 4. Runtime (`main.py`) — thường **không cần** đổi code

Hiện tại: đọc hai scalar từ `cfg` → `compose_env_from_ml_top2`. Chỉ cần cập nhật **số** trong config sau nghiên cứu, trừ khi sau này thêm tính năng khác (logging, v.v.).

---

## 5. Các file / vùng code chịu tác động

| File / vùng | Thay đổi |
|-------------|----------|
| `src/smartbinocular/config.py` | Chủ yếu **cập nhật giá trị** hai key khi chốt số. |
| `src/smartbinocular/main.py` | Thường **không đổi** cho story này. |
| `../../docs/tables/ml/…`, `THESIS_IMPROVEMENT_PLAN.md`, `DEPLOY_ML_MODEL.md` (cùng thư mục `legacy/md/`) | Từ phân tích 9 lớp → công thức tổng hợp → **một** cặp T1, T2. |
| `models/train_classifier.py`, `tools/compose_reliability_figure.py` | Mở rộng reliability + ECE + composite 9 lớp (§7). |
| (Tùy) sweep / notebook | Ứ viên theo lớp rồi **aggregate** (min/max/trung bình có trọng số — ghi rõ). |

**Không đổi:** `ml_inference.py`. **`train_classifier.py`:** đổi khi làm §7.

---

## 6. Từ phân tích 9 lớp → **một cặp ngưỡng toàn cục**

1. **Ưu tiên:** reliability + ECE **per class** (OOF, sau calibration) — §7.  
2. **Sweep / công thức:** có thể có bảng hoặc đường cong “ứng viên T1, T2” theo lớp; **tổng hợp** thành **một** `ml_confidence_threshold` và **một** `ml_secondary_confidence_threshold` bằng quy tắc đã ghi (ví dụ: conservative = lấy max ứng viên trên các lớp quan trọng; hoặc ràng buộc ECE tối đa).  
3. **Ba lớp đêm:** neo định tính + nằm trong bảng 9 lớp.  
4. **Lớp hiếm:** mẫu ít — ghi hạn chế; có thể **không** dùng lớp đó để siết toàn cục nếu ECE kém tin cậy.  
5. Ghi lại **một dòng công thức / quy tắc** trong thesis (từ N đồ thị → 2 số) để hội đồng thấy rõ, không cần map trong code.

---

## 7. Công cụ & đồ thị calibration cho **đủ 9 lớp** (không chỉ 3 lớp đêm)

**Hiện trạng:** pipeline training chỉ xuất reliability diagram + ECE cho ba lớp đêm:

- `models/train_classifier.py`: `plot_reliability_diagrams` lọc theo `_NIGHT_CLASS_NAMES = {night_clear, normal_night, nir_night}` — các cột OOF của lớp khác **không** được vẽ; metrics JSON lưu `ece_by_night_class`.
- `tools/compose_reliability_figure.py`: ghép cố định **3 panel** từ `{stem}_reliability_<class>.png` → ví dụ `docs/figures/ml/reliability/reliability_night_classes.png`.

Để **kết luận ngưỡng top-1 / top-2 có căn cứ** cho cả `normal_day`, `fog`, `rain`, `glare`, `backlight`, `transition` (và thống nhất với ba lớp đêm), cần **cùng một loại đồ thị** (reliability + ECE per-class từ OOF đã calibration) cho **mọi lớp trong `ENV_CLASSES`**, khi dữ liệu cho phép.

### 7.1. Mở rộng `plot_reliability_diagrams` (train time)

| Hạng mục | Đề xuất |
|----------|---------|
| Phạm vi lớp | Lặp trên **`ENV_CLASSES`** (hoặc tham số CLI `--reliability-scope night\|all` với `all` = đủ 9 tên). |
| Ngưỡng mẫu dương | Giữ tối thiểu (ví dụ ≥ 20 positives) để bin ổn định; lớp dưới ngưỡng → **skip plot + log cảnh báo** (hoặc plot với footnote “low n” trong luận văn). |
| Đầu ra file | Quy ước `{stem}_reliability_{class_name}.png`; bản lưu cho compose: [`docs/figures/ml/reliability`](../../docs/figures/ml/reliability). |
| Metrics JSON | Thêm hoặc đổi tên khóa **`ece_by_class`** (dict đủ các lớp đã tính được); có thể **giữ** `ece_by_night_class` tương thích ngược (subset 3 lớp) trong một thời gian. |

### 7.2. Mở rộng `tools/compose_reliability_figure.py`

- Tham số `--classes`: danh sách mặc định = **9 lớp** theo thứ tự `ENV_CLASSES` (hoặc đọc từ `feature_schema` nếu refactor nhỏ).
- Bố cục: **lưới** thay vì `1×3` — ví dụ **3×3** cho 9 panel; với số lớp ít hơn (thiếu PNG) → **bỏ qua ô** hoặc fail rõ ràng tùy flag `--strict`.
- Đầu ra gợi ý: `docs/figures/ml/reliability/reliability_all_env_classes.png` (giữ file 3-panel cũ cho slide chỉ tập trung đêm nếu cần).

### 7.3. Tài liệu & bảng số liệu

- Mở rộng `docs/tables/ml/ece_night_classes.md` thành bảng **tất cả lớp** (hoặc dùng `docs/tables/ml/ece_all_env_classes.md`) — nguồn: `ece_by_class` trong JSON bundle sau khi train.
- Cập nhật `THESIS_IMPROVEMENT_PLAN.md` / chương calibration: **9 đồ thị** (hoặc N lớp đủ mẫu) + quy tắc tổng hợp → **một cặp** `ml_confidence_threshold` / `ml_secondary_confidence_threshold` toàn cục.

### 7.4. Bổ sung tùy chọn (sau reliability one-vs-rest)

| Công cụ / ý tưởng | Mục đích cho ngưỡng |
|-------------------|---------------------|
| Script sweep offline (ví dụ `tools/sweep_ml_gate.py` hoặc notebook) | Grid search `threshold` trên vector OOF theo lớp → biểu đồ precision/recall hoặc tỷ lệ “ML active” vs sai lệch so với nhãn; chọn điểm elbow **per class**. |
| Calibration **có điều kiện cho top-2** | Khi `class_2 == k`, reliability của cột xác suất lớp `k` **trên mẫu có top-2 chứa k** — gần với rule 2 hơn one-vs-rest toàn tập; làm **sau** khi đã có plot 9 lớp cơ bản. |
| Báo cáo `calibration_curve` với `strategy="quantile"` | Lớp cực lệch mẫu: thử quantile bins để đọc đường cong ổn định hơn (ghi chú methodology trong thesis). |

**Thứ tự thực tế:** (1) vẽ đủ reliability + ECE 9 lớp từ OOF → (2) cập nhật bảng ECE + figure tổng hợp → (3) sweep ngưỡng / điều kiện top-2 nếu cần tinh hơn.

---

## 8. Telemetry (tùy chọn)

- Log **hai** giá trị global đang dùng (nếu cần đối chiếu với bảng thesis).  
- Histogram “gate top-1 theo lớp dự đoán” (optional) vẫn hữu ích để thấy lớp nào hay bị chặn, **không** cần map ngưỡng per-class.

---

## 9. Rủi ro

- **Một cặp toàn cục** có thể **không tối ưu** từng lớp hiếm — chấp nhận nếu ưu tiên đơn giản triển khai; luận văn nêu trade-off.  
- **Hành vi compositor:** rule 1–2 chỉ dùng hai float; rule 3–10 không đổi — regression `tests/test_env_compositor.py` khi đổi số.  
- **Gộp 9 lớp → 2 số:** quy tắc tổng hợp phải **nhất quán** (tránh lấy số từ một lớp mà bỏ qua lớp “căng” nhất).

---

## 10. Thứ tự triển khai gợi ý

1. **§7** — Mở rộng `train_classifier` / `compose_reliability_figure` + bảng ECE 9 lớp (bằng chứng).  
2. (Tùy) Sweep / notebook → ứng viên ngưỡng theo lớp; **aggregate** thành **một** T1, **một** T2.  
3. Cập nhật **hai** scalar trong `config.py` (và tài liệu deploy / thesis) — thường **không** cần sửa `main.py`.  
4. (Tùy) Ghi thêm cùng hai key trong preset RPi nếu cần số khác.  
5. `pytest tests/test_env_compositor.py` sau khi đổi số (hành vi cạnh biên theo `proba`).

---

## 11. Liên quan

- Reliability / ECE đêm: `docs/tables/ml/ece_night_classes.md`, `docs/figures/ml/reliability/reliability_night_classes.png`; sau mở rộng: bảng / figure **tất cả lớp** (§7).  
- Compositor: `compose_env_from_ml_top2` trong `env_presets.py`.  
- Taxonomy: `ENV_CLASSES` trong `feature_schema.py`.

---

*Tài liệu này: **phân tích / đồ thị đủ 9 lớp** để lập luận khoa học; **runtime** dùng **một cặp ngưỡng toàn cục** (top-1, top-2) — đơn giản cấu hình, không dict per-class, không deep merge. Triển khai tooling (§7) trước, sau đó chốt số vào CONFIG.*
