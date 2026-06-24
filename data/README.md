# data/ — Dữ liệu cục bộ (không đầy đủ trên git)

| Thư mục | Mục đích | Git |
| --- | --- | --- |
| **`weather/`** | Ảnh / dataset Kaggle cho ML offline (`mwd`, `weather11`, `darkface`, …); tham khảo thêm [Landscape image colorization](https://www.kaggle.com/datasets/theblackmamba31/landscape-image-colorization) trong `docs/DATASET_LOCAL_STATUS.md` | **Ignore** toàn bộ cây (xem root `.gitignore`) — tự tải và giải nén theo `docs/DATASET_LOCAL_STATUS.md` |
| **`thermal_optional/`** | LLVIP, KAIST, … (tùy chọn) | **Ignore** (cùng chính sách) |
| **`training/`** | JSONL đã merge / train-test split (`optical_only_train.jsonl`, …) | Thường **ignore** `*.jsonl` — không có placeholder trên git; tạo thư mục khi cần: `mkdir -p data/training` |

Chính sách ignore chi tiết: **`.gitignore`** ở root repo. Dataset và ảnh không nằm trên remote: sau clone, tạo `data/weather/...` / `data/thermal_optional/...` theo `docs/DATASET_LOCAL_STATUS.md` rồi giải nén dữ liệu cục bộ. Đánh giá domain-shift / tập công khai (LLVIP, KAIST, …): [`../docs/research_report_20260421_thesis_improvement.md`](../docs/research_report_20260421_thesis_improvement.md).
