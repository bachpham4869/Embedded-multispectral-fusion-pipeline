# Phase 2 — Soft gate & technical debt (pre–Phase 3)

**Recorded:** 2026-04-09  
**Purpose:** Ghi nhận baseline vận hành tối thiểu trên Raspberry Pi để **chấp nhận tạm** việc chuyển sang **Phase 3** (logging & data collection trong [`ML_HYBRID_SYSTEM_PLAN.md`](ML_HYBRID_SYSTEM_PLAN.md) cùng thư mục), thay cho exit criteria Phase 2 đầy đủ trong plan.

---

## 1. Điều kiện đã xác nhận (hands-on)

| Kiểm tra | Kết quả |
|----------|---------|
| Chạy pipeline trên RPi | Đã chạy |
| Chuyển các mode (trong phiên làm việc) | Đã thử, **không crash** |
| Soak 30 phút + log FPS/RAM/CPU theo plan | **Chưa** ghi nhận trong tài liệu này — xem mục 3 (nợ) |
| Profile hot path / `profile.out` | **Chưa** — xem mục 3 |

---

## 2. Soft gate — định nghĩa đã thống nhất

- **Soft gate Phase 2** = pipeline **ổn định đủ** để bắt đầu Phase 3 (MLLogger, 23-feature JSONL, tools label/check), với giả định: regression hiệu năng sẽ được theo dõi khi bật logging.
- **Không** thay thế **exit criteria Phase 2 đầy đủ** trong [`ML_HYBRID_SYSTEM_PLAN.md`](ML_HYBRID_SYSTEM_PLAN.md) §7 (soak 30 min, RSS, CPU ≤75%, `logs/performance/`, cProfile, v.v.) nếu sau này cần báo cáo formal hoặc so sánh ML trước/sau một cách cứng.

---

## 3. Nợ kỹ thuật (ưu tiên xử lý song song hoặc ngay sau Phase 3.1)

Theo plan Phase 2 và rà soát repo:

| ID | Hạng mục | Ghi chú |
|----|----------|---------|
| **D1** | Soak **≥30 phút** NIR + thermal, FPS ≥20, RSS ổn, CPU, **zero crash** | Bổ sung log `logs/performance/perf_*.jsonl` hoặc tương đương khi có thể. |
| **D2** | **cProfile** 2 phút trên RPi, phân tích **top-3** hàm theo thời gian tích lũy | Mục tiêu plan: không có hàm đơn lẻ >20 ms/frame (hướng dẫn, không cứng). |
| **D3** | **FrameQualityScore (R10)** trên capture path (`utils` / capture) | Trong plan Phase 2.7 — chưa có trong codebase tại thời điểm soft gate. |
| **D4** | Verify **single glare path** (`display_pipeline`) + **deterministic mode switching** (Fix 6) | Plan 2.4 / 2.5 — nên có checklist test ngắn hoặc ghi chú phiên bản đã verify. |
| **D5** | **Sequence voting R9** vs hysteresis hiện tại | `EnvPresetController` đã có debounce/hysteresis; nếu cần đúng spec R9 đầy đủ, đánh giá lại sau khi có dữ liệu ENV. |
| **D6** | Git tag **`v2.0-stable-pre-ml`** (plan 2.8) | Tùy repo; nếu dùng git, tag tại commit baseline sau soak/profile. |
| **D7** | Đồng bộ **ML plan** với code (ví dụ Phase 1 checklist vs NIR/CLAHE) | Tránh hiểu nhầm khi đọc lại tài liệu cũ. |

---

## 4. Khuyến nghị khi bước vào Phase 3

1. Ghi **commit hash** hoặc mô tả bản build RPi vào commit message / issue khi bật `ML_LOG_*`.
2. Giữ **MLLogger** không chặn hot path (buffer/async, flush định kỳ) như plan.
3. Sau **MLLogger** chạy ổn trên Pi ~10 phút, cân nhắc **D1 + D2** để “đóng cứng” baseline trước khi train model (Phase 4).

---

## 5. Liên kết

- [`ML_HYBRID_SYSTEM_PLAN.md`](ML_HYBRID_SYSTEM_PLAN.md) — Phase 2 (§7), Phase 3 (logging).
- `CLAUDE.md` — layout module, hook ML tương lai.
