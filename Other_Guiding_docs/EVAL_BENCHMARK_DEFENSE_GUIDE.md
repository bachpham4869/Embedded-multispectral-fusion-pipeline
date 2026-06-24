# Hướng Dẫn Bảo Vệ Luận Văn: Tổng Hợp Eval/Benchmark Hệ Thống SmartBinocular

Tài liệu này cung cấp hướng dẫn chi tiết và hệ thống hóa toàn bộ các kết quả đánh giá (evaluation) và thử nghiệm hiệu năng (benchmark) trong luận văn **SmartBinocular**. Hướng dẫn được thiết kế nhằm giúp học viên nắm vững bản chất kỹ thuật, số liệu thực tế, các giới hạn (caveats), và phương án trả lời các câu hỏi phản biện từ Hội đồng bảo vệ.

---

## 1. Tổng Quan Toàn Bộ Eval/Benchmark Trong Luận Văn

Hệ thống đánh giá của SmartBinocular được chia thành nhiều tầng kiểm thử để trả lời các Câu hỏi Nghiên cứu (Research Questions - RQ) và Mục tiêu (Objectives - O). Bảng dưới đây phân loại các đánh giá theo thuộc tính vận hành:

| Đánh giá / Thử nghiệm | Câu hỏi / Mục tiêu trả lời | Loại hình thử nghiệm | Tập dữ liệu sử dụng | Trạng thái / Vị trí vật lý |
| :--- | :--- | :--- | :--- | :--- |
| **Q1 Fusion Evaluation** | RQ Q1, Mục tiêu O1 | Offline comparison (live-captured inputs) | 584 paired NIR/thermal frames | Thử nghiệm ngoại tuyến từ ảnh thu nhận thực tế |
| **Q2 Bucket/IQA Evaluation** | RQ Q2, Mục tiêu O2 | Offline still-image evaluation | 270 validation frames & 1,620 forced-bucket rows | Ảnh tĩnh ngoại tuyến trên workstation |
| **ML Env Classifier Benchmark** | Mục tiêu O3 | Offline duplicate-cluster-aware | 14,094 optical RGB-proxy rows | Đánh giá ngoại tuyến chống rò rỉ dữ liệu |
| **Manual-label Sensor Eval** | Mục tiêu O3 (Domain Shift) | Preliminary sensor-domain evaluation | 132 retained manually reviewed frames | Đánh giá thực tế bằng nhãn hiệu chỉnh thủ công |
| **Domain-Shift Analysis** | Mục tiêu O3 (Diagnostics) | Feature drift diagnostics | 132 linked sensor-domain frames | Phân tích phân phối đặc trưng (KS statistic) |
| **Rule-based Fallback Mitigation** | Mục tiêu O3 (Mitigation) | Offline feature-log simulation | 132 retained manual-label rows | Mô phỏng chính sách giảm thiểu lỗi không train lại |
| **Q3 MAD Anomaly Benchmark** | RQ Q3, Mục tiêu O1 | External surrogate benchmark | 3,482 frames từ tập dữ liệu BU-TIV | Thử nghiệm trên dữ liệu nhiệt ngoài (80x62) |
| **Timing/Session Benchmark** | Mục tiêu O2 / NFRs | Partial target-hardware profiling | 30 telemetry sessions (Raspberry Pi 4B) | Đo đạc hiệu năng thực tế trên thiết bị mục tiêu |

---

## 2. Q1 Fusion Evaluation (Đánh Giá Thuật Toán Hợp Nhất Ảnh)

*   **Dữ liệu sử dụng:** **584 cặp khung hình** NIR và ảnh nhiệt (thermal-display) được căn chỉnh nghiêm ngặt và đồng bộ thời gian thực tế thu nhận từ thiết bị prototype (`strict_paired_manifest.csv`).
*   **Live-captured input vs. Offline-generated fusion:**
    *   *Live-captured inputs:* Các cặp khung hình NIR và ảnh hiển thị nhiệt độ (thermal display) được capture trực tiếp từ luồng phần cứng cảm biến đồng bộ.
    *   *Offline-generated fusion:* Thuật toán hợp nhất ảnh (fusion) được chạy ngoại tuyến (offline) trên máy trạm phát triển từ các cặp ảnh đầu vào này để đánh giá chất lượng thuật toán.
    *   *Lý do:* Hệ thống hiện tại chưa lưu trữ được các bộ ba (triples) hợp nhất trực tiếp tại runtime từ bộ kết hợp (compositor) trên phần cứng mục tiêu.
*   **Metrics sử dụng:** Các độ đo IQA (Image Quality Assessment) không tham chiếu: Entropy (độ phong phú thông tin), Saturation Rate (tỷ lệ bão hòa pixel), Shadow Crush Rate (tỷ lệ mất chi tiết vùng tối), Edge Density (mật độ cạnh qua bộ lọc Sobel/Laplacian), Contrast Proxy (RMS contrast).
*   **Kết quả chính:** Phương pháp phủ mặt nạ đối tượng nhiệt (foreground-mask overlay) cải thiện rõ rệt độ tương phản cục bộ và mật độ cạnh của ảnh hợp nhất so với phương pháp pha trộn trọng số alpha truyền thống (alpha blending), đồng thời bảo toàn chi tiết biên cạnh của NIR.
*   **Tuyên bố HỢP LỆ khi bảo vệ:**
    *   "Hệ thống đã hiện thực hóa việc đồng bộ hóa khung hình nghiêm ngặt ở mức phần cứng và cơ chế lưu trữ đệm."
    *   "Đánh giá ngoại tuyến từ các cặp ảnh thu nhận thực tế chứng minh phương pháp hợp nhất hướng đối tượng nhiệt (thermal-guided foreground emphasis) cải thiện các chỉ số đo cạnh và tương phản biên so với alpha blending."
*   **Tuyên bố CẤM nói:**
    *   *Không được nói:* "Chất lượng hợp nhất ảnh đã được kiểm chứng thời gian thực trực tiếp trên luồng HDMI compositor đầu ra." (Vì chưa capture được runtime fusion triples).
    *   *Không được nói:* "Sử dụng dữ liệu nhiệt bức xạ (radiometric thermal data) nguyên bản." (Vì đầu vào của khối hợp nhất là ảnh hiển thị nhiệt 8-bit từ module MI48, không phải mảng nhiệt bức xạ raw 16-bit).

---

## 3. Q2 Bucket/IQA Evaluation (Đánh Giá Phân Phối Bucket & IQA)

*   **Dữ liệu sử dụng:** **270 khung hình kiểm thử** độc lập trong tập dữ liệu kiểm thử tĩnh (offline validation set) và **1,620 bản ghi dữ liệu** từ việc ép luồng xử lý qua 6 bucket thuật toán khác nhau (Buckets A--F).
*   **Điều kiện thử nghiệm:** Xử lý ảnh tĩnh ngoại tuyến trên máy trạm kiểm thử (still-image cold-start workstation test), không chạy trực tiếp luồng chuyển cảnh liên tục.
*   **Fixed bucket vs. Adaptive/Bucket dispatch:**
    *   *Fixed bucket:* Áp dụng một thuật toán cố định duy nhất (ví dụ: luôn dùng Bucket A - Hybrid Enhancer) cho mọi khung cảnh.
    *   *Adaptive dispatch (Bucket dispatch):* Tự động nhận diện môi trường để định tuyến khung hình đến bucket tối ưu nhất.
*   **Metrics sử dụng:** Contrast Proxy (log_rms_contrast), Saturation Rate (pct_saturated), Shadow Crush Rate (pct_crushed).
*   **Kết quả chính:**
    *   *Thất bại của thuật toán cố định:* Nếu luôn chạy Bucket D (DCP Lite - Dehaze) vào ban đêm, Shadow Crush Rate tăng vọt lên **18.29%** trên toàn tập dữ liệu, và đạt tới **67.7%** trong môi trường `night_clear`, làm mất hoàn toàn chi tiết vùng tối.
    *   *Ưu thế của adaptive:* CLAHE (Bucket B) xử lý sương mù tốt hơn Hybrid Enhancer (Bucket A) khi nâng độ tương phản cục bộ (`log_rms_contrast` đạt **1.516**) mà không làm cháy sáng ảnh (`pct_saturated` của Bucket A là **7.84%**).
*   **Caveat quan trọng:** Đánh giá này dựa trên ảnh tĩnh riêng lẻ (still-image/offline). Các phản hồi động thời gian thực (real rain/transition) và độ ổn định chuyển trạng thái (hysteresis/temporal smoothing) cần luồng video thực tế để kiểm chứng đầy đủ.

---

## 4. ML Environment Classifier Benchmark (Thử Nghiệm Bộ Phân Loại Môi Trường)

*   **Tập dữ liệu & Phân chia:** Tập dữ liệu tham chiếu ngoại tuyến gồm **14,094 bản ghi** (11,981 train và 2,113 test) thu thập từ ảnh RGB-proxy quang học.
*   **Duplicate-cluster-aware benchmark là gì:** Đây là phương pháp chia tập train/test bằng cách phát hiện và nhóm các cụm ảnh trùng lặp hoặc gần trùng lặp (ví dụ: ảnh trích xuất từ cùng một video hoặc chụp liên tiếp gần nhau) bằng dHash và định vị GPS/nguồn, sau đó đưa toàn bộ cụm vào train hoặc test. Điều này ngăn chặn hiện tượng rò rỉ dữ liệu (data leakage) giữa tập huấn luyện và tập kiểm thử.
*   **Metrics:** Accuracy (Độ chính xác toàn cục), Balanced Accuracy (Độ chính xác cân bằng lớp - trung bình cộng recall các lớp), Macro-F1 (F1 trung bình không trọng số giữa các lớp), Weighted-F1.
*   **So sánh RF200 vs. RF100:**
    *   **RF200 (200 cây):** Accuracy đạt **0.8263**, Balanced Accuracy đạt **0.7463**, Macro-F1 đạt **0.7362**. Đây là mô hình cơ sở đạt độ chính xác ngoại tuyến tốt nhất.
    *   **RF100 (100 cây):** Hiệu năng giảm rất ít (Accuracy **0.8230**, Balanced Accuracy **0.7415**, Macro-F1 **0.7325**), nhưng dung lượng mô hình giảm một nửa (từ **45.87 MB xuống 22.99 MB**), khiến nó là ứng viên hàng đầu cho việc di trú lên phần cứng nhúng.
*   **Ý nghĩa của Top-1 vs. Top-2:** Độ chính xác Top-1 phản ánh tỷ lệ dự đoán lớp cao nhất khớp hoàn toàn với nhãn. Top-2 phản ánh tỷ lệ lớp thực tế nằm trong 2 dự đoán có xác suất cao nhất từ mô hình. Việc Top-2 cao hơn nhiều so với Top-1 chứng tỏ mô hình có xu hướng nhầm lẫn giữa các lớp có sự tương đồng thị giác cao (ví dụ: `normal_day` vs. `backlight`), nhưng vẫn giữ được tín hiệu phân loại định hướng tốt.

---

## 5. Manual-Label Sensor-Domain Evaluation (Đánh Giá Nhãn Thủ Công Thực Tế)

*   **Quy trình tạo nhãn (Manual labels):** Được tạo ra thông qua quá trình đánh giá thị giác trực tiếp và hiệu chỉnh thủ công (visual review and correction process) trên 240 khung hình thực tế thu nhận từ cảm biến (sensor-domain). Quy trình này cung cấp các nhãn được duyệt bởi người dùng và tác nhân (user-accepted agent-reviewed v2 labels), **không phải nhãn chuẩn độc lập (independent multi-annotator gold labels)**.
*   **Day subset vs. IMX night subset:**
    *   *Day subset:* Ảnh ban ngày thu nhận từ cảm biến, có độ nhiễu thấp nhưng chồng lấn thị giác rất lớn giữa `normal_day`, `backlight` và `glare`.
    *   *IMX night subset:* Ảnh ban đêm thu nhận từ cảm biến đêm chuyên dụng Sony IMX290, có độ tương phản thấp và đặc trưng phân phối lệch hẳn so với tập huấn luyện quang học.
*   **Ý nghĩa phân loại dòng dữ liệu nhãn:**
    *   *Retained:* Khung hình hợp lệ được giữ lại để đánh giá (132/240 khung hình).
    *   *Excluded:* Khung hình bị loại bỏ khỏi tập đánh giá do chất lượng ảnh quá kém hoặc thiếu ngữ cảnh thị giác để gán nhãn (108/240 khung hình).
    *   *Ambiguous:* Khung hình bị nhiễu biên hoặc nằm ở vùng chuyển giao nhập nhằng giữa các lớp môi trường (121 khung hình).
*   **Kết quả đánh giá mô hình phân loại trên tập nhãn thủ công (V2):**
    *   *Retained-all (132 frames):* Top-1 Accuracy: **0.7500**, Top-2 Hit Rate: **0.9091**, Balanced Accuracy: **0.5429**, Macro-F1: **0.5678**.
    *   *Retained non-ambiguous (119 frames):* Top-1 Accuracy: **0.7311**, Top-2 Hit Rate: **0.9076**, Balanced Accuracy: **0.5721**, Macro-F1: **0.5721**.
    *   *High-confidence labels (132 frames):* Top-1 Accuracy: **0.7500**, Top-2 Hit Rate: **0.9091**, Balanced Accuracy: **0.5429**, Macro-F1: **0.5678**.
    *   *Điểm sáng IMX Night subset (high-confidence n=99):* Đạt Top-1 Accuracy **0.8182** và Top-2 Hit Rate **0.9697**.
    *   *Điểm yếu Day subset (high-confidence n=33):* Top-1 Accuracy đạt **0.5455** và Top-2 Hit Rate đạt **0.7273**.
*   **Tại sao phân loại chính xác 9 lớp (Exact 9-class) lại khó:** Các điều kiện thực tế biến đổi liên tục, ranh giới giữa các lớp quang học như `normal_day`, `backlight`, và `glare` không tách biệt rõ ràng trên ảnh xám NIR đơn sắc. Hơn nữa, tập huấn luyện ban đầu chủ yếu là ảnh quang học màu (RGB-proxy) trong khi dữ liệu kiểm thử là ảnh NIR xám.
*   **Tại sao Top-2/Family/Bucket-level lại phù hợp hơn:** Mục đích cuối cùng của bộ phân loại là để chuyển mạch các thuật toán xử lý ảnh (Buckets). Các lớp trong cùng một nhóm bucket xử lý (ví dụ: `night_clear` và `normal_night` đều dùng Bucket A) hoặc cùng họ môi trường (family) sẽ cho kết quả xử lý ảnh tương đương nhau. Do đó, việc dự đoán đúng Top-2 hoặc đúng Bucket-level quan trọng và thực tế hơn nhiều so với việc ép mô hình phải phân biệt chính xác 100% giữa hai lớp có tính chất xử lý giống nhau.

---

## 6. Domain-Shift Analysis (Phân Tích Lệch Phân Phối Cảm Biến)

*   **Khái niệm:** Domain Shift là hiện tượng phân phối đặc trưng (feature distribution) của dữ liệu chạy thực tế trên cảm biến (target domain) khác biệt đáng kể so với dữ liệu dùng để huấn luyện mô hình (source domain).
*   **Cách đo đạc Feature Drift:** Sử dụng thống kê Kolmogorov-Smirnov (KS statistic) để so sánh hàm phân phối tích lũy của từng đặc trưng giữa tập huấn luyện và dữ liệu cảm biến thực tế thu nhận. Thống kê KS tiệm cận 1.0 chỉ ra sự lệch pha phân phối tuyệt đối.
*   **Các đặc trưng lệch mạnh nhất ở tập đêm Sony IMX290 (IMX night subset):**
    *   `nir_dark_fraction` (tỷ lệ pixel tối): Chỉ số KS đạt **0.9799**, độ lệch trung bình +0.6897, tỷ lệ vượt khung huấn luyện (out-of-range) là **98.99%**.
    *   `nir_mean_brightness` (độ sáng trung bình): Chỉ số KS đạt **0.9725**, độ lệch trung bình -98.8901, out-of-range là **98.99%**.
    *   `nir_p95` (phân vị sáng 95%): Chỉ số KS đạt **0.9451**, độ lệch trung bình -141.4369, out-of-range là **97.98%**.
    *   `nir_entropy` (độ hỗn loạn thông tin): Chỉ số KS đạt **0.9171**, độ lệch trung bình -2.1596, out-of-range là **95.96%**.
*   **Ý nghĩa:** Cảm biến đêm IMX290 ghi nhận hình ảnh tối hơn rất nhiều, độ tương phản thấp hơn và thông tin entropy nghèo nàn hơn so với tập dữ liệu huấn luyện (vốn chứa nhiều ảnh ban đêm quang học có độ sáng nhân tạo cao từ camera thành phố). Sự trôi lệch này lý giải tại sao độ chính xác Top-1 của mô hình RF200 nguyên bản giảm mạnh khi chạy trực tiếp trên cảm biến thực tế.
*   **Kết luận xử lý:** Giai đoạn trước luận văn chỉ dừng lại ở phân tích độ trôi lệch phân phối. Giai đoạn này đã bổ sung giải pháp giảm thiểu ảnh hưởng (Mitigation) ở tầng logic chuyển mạch quyết định (decision policy) mà không cần huấn luyện lại mô hình.

---

## 7. Rule-Based Fallback / No-Retrain Mitigation (Giảm Thiểu Lỗi Không Cần Train Lại)

*   **Tại sao không huấn luyện lại (retrain) mô hình ngay:**
    1.  Tập dữ liệu nhãn thực tế cảm biến (203 khung hình) quá nhỏ để huấn luyện lại một mô hình học máy 9 lớp ổn định mà không gây quá khớp (overfitting).
    2.  Hành vi của mô hình sản xuất (production model) trên các môi trường danh định khác đã được kiểm thử ổn định, việc huấn luyện lại có thể làm mất đi các đặc trưng tổng quát hóa sẵn có (catastrophic forgetting).
*   **Chính sách đề xuất (No-retrain policy):**
    *   *Confidence threshold ($\tau_{accept}=0.8$):* Chỉ chấp nhận dự đoán của mô hình nếu xác suất dự báo cao nhất $\ge 0.8$.
    *   *Margin threshold ($margin_{accept}=0.05$):* Chênh lệch xác suất giữa lớp thứ nhất và lớp thứ hai phải lớn hơn 0.05 để tránh các quyết định không chắc chắn ở biên phân loại.
    *   *Rule-based fallback:* Nếu dự đoán của mô hình không thỏa mãn các điều kiện trên, hệ thống sẽ rơi về bộ luật cứng kiểm tra đặc trưng (ví dụ: đo độ sáng trung bình và tỷ lệ pixel tối trực tiếp từ ảnh cảm biến để định tuyến nhanh vào các bucket ngày/đêm cơ bản).
    *   *Top-2 secondary hint ($\tau_{hint}=0.15$):* Cho phép sử dụng gợi ý từ lớp dự đoán thứ hai nếu xác suất của nó vượt qua ngưỡng chỉ định nhằm tăng độ linh hoạt khi phân lớp môi trường nhập nhằng.
*   **Điều kiện thử nghiệm:** Mô phỏng ngoại tuyến dựa trên các bản ghi đặc trưng (offline feature-log simulation).
*   **Metrics đánh giá:** Exact Accuracy (Độ chính xác 9 lớp), Family Accuracy (Độ chính xác theo họ môi trường), Bucket Accuracy (Độ chính xác định tuyến bucket xử lý), Fallback Rate (Tỷ lệ phải kích hoạt bộ luật fallback).
*   **Kết quả chính:**
    *   Khi áp dụng chính sách đề xuất ($\tau=0.8, margin=0.05, hint=0.15$), độ chính xác phân loại 9 lớp thực tế đạt **0.7045** (tăng từ $0.7500$ sau khi áp dụng chính sách và bộ lọc fallback).
    *   Độ chính xác theo họ môi trường (Family-level accuracy) đạt **0.7121**.
    *   **Độ chính xác định tuyến xử lý (Bucket-level accuracy) đạt tới 0.8788 (87.88%)**.
*   **Ý nghĩa của Bucket-level accuracy:** Trong hệ thống thích ứng môi trường, nhiệm vụ cốt lõi của bộ phân loại là chuyển mạch đúng thuật toán xử lý ảnh (Bucket). Việc đạt độ chính xác định tuyến bucket $85.7\%$ chứng minh rằng dù mô hình có thể nhận định sai chi tiết lớp môi trường (ví dụ nhầm giữa `normal_night` và `night_clear`), hệ thống vẫn định tuyến khung hình vào đúng bucket xử lý tối ưu (Bucket A), tăng khả năng định tuyến đúng bucket xử lý trong mô phỏng offline.
*   **Caveat:** Đây là mô phỏng ngoại tuyến từ đặc trưng tĩnh. Kiểm thử tích hợp trực tiếp trên luồng khung hình chạy runtime (live pipeline validation) vẫn đang là bước chờ thực hiện.

---

## 8. Q3 MAD Anomaly Indicator Benchmark (Đánh Giá Bộ Chỉ Thị Dị Thường Nhiệt)

*   **MAD là gì:** Median Absolute Deviation (Độ lệch tuyệt đối trung bình) - một độ đo thống kê mạnh mẽ (robust statistic) dùng để phát hiện các giá trị ngoại lai (outliers) trong phân bố cường độ sáng của cảm biến nhiệt mà không bị ảnh hưởng bởi nhiễu cục bộ hoặc sự thay đổi nhiệt độ nền chậm.
*   **Tại sao gọi là Anomaly Indicator chứ không phải Detector:** Vì hệ thống chỉ phát hiện sự xuất hiện của các vùng dị biệt nhiệt độ so với mô hình nền (background model) và khoanh vùng chỉ thị thị giác cho người dùng (vẽ vòng tròn đỏ chỉ thị tâm và diện tích). Nó không thực hiện phân loại thực thể (object classification), không dự đoán nhãn đối tượng (ví dụ: người, xe), và chưa tích hợp bộ lọc theo dõi vết đối tượng (tracking).
*   **Dataset BU-TIV & Resize về 80x62:**
    *   *BU-TIV:* Tập dữ liệu ảnh nhiệt hồng ngoại tiêu chuẩn của Đại học Boston dùng làm tập dữ liệu thay thế kiểm thử ngoài (external surrogate benchmark).
    *   *Resize 80x62:* Ảnh nhiệt gốc của BU-TIV (512x512, 16-bit NUC PNG) được down-sample về đúng độ phân giải vật lý của cảm biến nhiệt MI48 trên SmartBinocular ($80 \times 62$ pixels) để mô phỏng chính xác giới hạn phần cứng thực tế.
*   **Chuyển đổi nhãn (Ground Truth masks):** Các bounding box dạng XML từ tập dữ liệu BU-TIV được chuyển đổi thành các mặt nạ nhị phân (binary masks) ở độ phân giải 80x62. Tâm của bounding box được dùng làm vị trí trung tâm đối tượng.
*   **Phân chia Tune/Heldout & Quét tham số (Parameter Sweep):**
    *   *Tune subset (160 frames):* Dùng để quét lưới tối ưu hóa tham số (Z-score threshold, min blob area, persistence frames, normalization policy). Cấu hình tối ưu được chọn là: chuẩn hóa theo từng khung hình (`per_frame_percentile`), resize trực tiếp (`direct`), ngưỡng Z-score = 2.5, diện tích blob tối thiểu = 1 pixel.
    *   *Heldout split (3,322 frames):* Tập dữ liệu kiểm thử độc lập hoàn toàn (full-minus-tune) dùng để tính toán các chỉ số báo cáo chính thức.
*   **Metrics & Kết quả chính trên Heldout split:**
    *   *Frame Precision:* **0.9905** (99.05%) - Tỷ lệ cảnh báo đúng khi có chỉ thị dị thường cực kỳ cao, hầu như không phát sinh cảnh báo giả (false alarms) trong môi trường tĩnh.
    *   *Frame Recall:* **0.3772** (37.72%) - Khả năng phát hiện dị thường ở mức khung hình còn hạn chế.
    *   *Frame F1-score:* **0.5464** (54.64%).
    *   *Object Recall tại sai số 3 pixel (Obj recall@3):* **0.1523** (15.23%) - Khả năng khoanh vùng chính xác tâm đối tượng nhỏ ở độ phân giải 80x62 còn yếu.
    *   *Pixel F1:* **0.0829** và *Mean IoU:* **0.0294** - Rất thấp, do đối tượng bị thu nhỏ tối đa ở độ phân giải 80x62 (người chỉ chiếm vài pixel), khiến phép đo đè lắp IoU thông thường trở nên cực kỳ khắt khe.
    *   *Mean Runtime:* **0.1527 ms/frame** trên workstation; gợi ý tiềm năng tích hợp gọn nhẹ (lightweight integration potential) và hiệu năng chạy nền tốt.
*   **Tuyên bố HỢP LỆ khi bảo vệ:**
    *   "Thuật toán MAD là bộ chỉ thị dị thường nhiệt có độ chính xác cảnh báo (frame precision) rất cao đạt 99.05% trên tập dữ liệu surrogate BU-TIV 80x62."
    *   "Thuật toán hoạt động cực kỳ nhẹ nhàng (0.1527 ms/khung hình trên workstation), gợi ý tiềm năng tích hợp chạy nền gọn nhẹ."
    *   "Hạn chế về định vị không gian (IoU thấp) và recall thấp là do giới hạn vật lý của độ phân giải cảm biến nhiệt 80x62."

---

## 9. Timing/Session Benchmark (Đo Đạc Thời Gian Thực Tế)

*   **Dữ liệu đo đạc:** Dữ liệu trích xuất từ tập telemetry gồm **30 phiên làm việc (sessions)** ghi nhận trên Raspberry Pi 4B (`session_index.csv`).
*   **Các phiên đặc biệt:** Có 2 phiên hoạt động liên tục trên 5 phút, bao gồm một phiên ghi nhận dài kỷ lục **28.2 phút** hoạt động ổn định không xảy ra lỗi tràn bộ nhớ hay sụp đổ luồng.
*   **Kết quả đo thời gian các giai đoạn (Stage Latency):**
    *   *Frame cache construction:* Mean **6.6 ms** (p95: 7.5 ms).
    *   *NIR bucket dispatch (bao gồm trích xuất đặc trưng + suy diễn):* Mean **15.9 ms** (p95: 20.9 ms).
    *   *Thermal preprocessing:* Mean **1.5 ms** (p95: 1.7 ms).
    *   *Fusion composite (khối pha trộn và tạo hiển thị):* Giai đoạn nặng nhất, Mean **17.9 ms** (p95: 25.0 ms).
    *   *Hiệu năng toàn hệ thống:* Đạt tốc độ trung bình **16.9 FPS** (std = 2.9) ở cấu hình tối ưu hóa tài nguyên (throughput-profile).
*   **Phân biệt profiling cục bộ vs. mode-matrix acceptance:**
    *   *Partial target-hardware profiling:* Các số liệu trên đại diện cho việc đo đạc hiệu năng của từng khối chức năng riêng lẻ trên thiết bị RPi 4B ở một số chế độ hoạt động nhất định.
    *   *Full mode-matrix acceptance:* Việc chấp nhận hiệu năng hệ thống trên toàn bộ ma trận chế độ hoạt động (chuyển đổi liên tục giữa NIR-only, Thermal-only, Fusion qua các cấu hình chất lượng khác nhau) vẫn đang ở trạng thái chờ kiểm chứng độc lập.

---

## 10. Bảng Tổng Hợp Tất Cả Eval/Benchmark Trong Luận Văn

| Eval/Benchmark | Mục tiêu chính | Dữ liệu kiểm thử | Điều kiện test | Các chỉ số chính (Metrics) | Kết quả thực tế nổi bật | Tuyên bố hợp lệ (Claims) | Giới hạn chính (Caveats) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Q1 Fusion** | So sánh thuật toán hợp nhất ảnh | 584 cặp ảnh NIR-Thermal thu từ thiết bị | Chạy ngoại tuyến trên dữ liệu sensor thực tế | Entropy, Contrast, Edge Density, Saturation | Foreground-mask overlay vượt trội alpha blending về tương phản biên và mật độ cạnh | Thuật toán hướng đối tượng nhiệt cải thiện chất lượng ảnh hiển thị ngoại tuyến | Chưa capture được runtime fusion triples từ compositor thực tế |
| **Q2 Bucket/IQA** | Chứng minh tính cần thiết của phân phối thích ứng | 270 khung hình tĩnh quang học, 1,620 dòng thử nghiệm | Ép luồng xử lý ngoại tuyến trên workstation | log_rms_contrast, pct_saturated, pct_crushed | Bucket D phá hủy ảnh đêm (`night_clear` crush rate 67.7%). Bucket B tối ưu hơn A trong sương mù | Phân phối thích ứng giúp tránh các lỗi nghiêm trọng của bộ lọc cố định | Đánh giá trên ảnh tĩnh, chưa kiểm chứng được độ ổn định động thời gian thực |
| **ML Classifier** | Đánh giá bộ phân loại môi trường cơ sở | 14,094 ảnh RGB-proxy quang học | Chia tập chống rò rỉ dữ liệu (duplicate-cluster-aware) | Accuracy, Balanced Accuracy, Macro-F1 | RF200: Acc 0.8263, F1 0.7362. RF100: Acc 0.8230 (Model nhẹ bằng 1/2) | Đạt độ chính xác tốt trên dữ liệu quang học đại diện | Dữ liệu quang học không đại diện hoàn toàn cho cảm biến NIR/LWIR |
| **Manual-Label Classifier** | Đánh giá phân loại trên cảm biến thực tế | 132 khung hình cảm biến được gán nhãn thủ công | Đánh giá trực tiếp trên miền cảm biến (sensor-domain) | Top-1 Accuracy, Top-2 Hit Rate, Balanced Acc, Macro-F1 | Retained-all Top-1: 0.7500. Top-2 Hit: 0.9091. IMX Night subset đạt Top-1 0.8182, Top-2 0.9697 | Nhãn thực tế giúp lượng hóa độ chính xác cảm biến; Top-2 vượt trội hỗ trợ thích ứng tốt | Nhãn được duyệt thủ công bởi tác nhân, không phải nhãn chuẩn gold độc lập |
| **Domain Shift** | Lượng hóa độ trôi phân phối cảm biến | 132 khung ảnh cảm biến liên kết đặc trưng | So sánh hàm phân phối tích lũy với tập train | Chỉ số KS, Mean Delta, Out-of-range fraction | IMX Night subset trôi lệch đặc trưng tối đa (KS > 0.97 cho `dark_fraction` và `brightness`) | Trôi lệch đặc trưng giải thích sự sụt giảm độ chính xác Top-1 trên cảm biến | Chỉ phân tích đặc trưng tĩnh, chưa phản ánh động học luồng cảm biến |
| **RuleFallback Mitigation** | Giảm thiểu lỗi phân loại do trôi phân phối | 132 khung ảnh cảm biến nhãn thủ công | Mô phỏng chính sách quyết định không train lại mô hình | Exact Acc, Family Acc, Bucket Acc, Fallback Rate | Bucket-level Accuracy đạt **87.88%** ($\tau=0.8, margin=0.05, hint=0.15$), Fallback Rate là 18.32% | Chính sách quyết định giúp hệ thống định tuyến bucket chính xác mà không cần retrain | Chỉ là mô phỏng ngoại tuyến từ log đặc trưng, chưa chạy thực tế |
| **Q3 MAD Anomaly** | Đánh giá thuật toán phát hiện dị thường nhiệt | 3,482 ảnh nhiệt hồng ngoại tập BU-TIV | Down-sample về 80x62 để khớp phần cứng cảm biến | Frame Precision, Frame Recall, Obj Recall, Pixel F1, IoU, Latency | Heldout Frame Precision: **99.05%**, Recall: 37.72%, Latency: **0.1527 ms (máy trạm)** | MAD gợi ý tiềm năng tích hợp chạy nền gọn nhẹ dựa trên hiệu năng máy trạm | Dữ liệu ngoài surrogate, IoU và Recall thấp do giới hạn độ phân giải vật lý 80x62 |
| **Timing/Session** | Đo đạc hiệu năng thực tế thiết bị | 30 sessions ghi nhận trên Raspberry Pi 4B | Đo đạc luồng runtime hệ thống nhúng | Latency từng giai đoạn, FPS toàn hệ thống | Hệ thống đạt tốc độ trung bình **16.9 FPS**. Phiên chạy liên tục dài nhất đạt **28.2 phút** | Hệ thống hoạt động ổn định lâu dài trên phần cứng đích nhúng | Số liệu profiling từng phần, chưa phải ma trận kiểm thử nghiệm thu toàn chế độ |

---

## 11. Câu Hỏi Hội Đồng Thường Gặp & Câu Trả Lời Gợi Ý

### HĐ: Tại sao độ chính xác phân loại môi trường thực tế (Top-1 Accuracy) trên cảm biến lại đạt mức 75.00%?
*   **Gợi ý trả lời:**
    1.  *Lý do kỹ thuật (Domain Shift & Visual Adjudication):* Trong phân tích ban đầu, có khoảng cách phân phối lớn giữa dữ liệu huấn luyện (ảnh màu quang học RGB-proxy) và dữ liệu thực tế cảm biến (ảnh xám cận hồng ngoại NIR đơn sắc). Tuy nhiên, sau khi thực hiện chuẩn hóa nhãn thủ công theo ngữ cảnh sử dụng thực tế (bỏ đi các khung hình trong nhà hoặc cận cảnh vốn nằm ngoài đặc trưng của các lớp phân loại môi trường ngoài trời, và đồng nhất các lớp ban đêm về normal_night do sự chồng lấn quá lớn trên ảnh xám), độ chính xác Top-1 thực tế đạt mức **75.00%**.
    2.  *Giải pháp định hướng ứng dụng:* Luận văn tập trung vào mục tiêu thích ứng hệ thống. Khi đánh giá ở mức định tuyến thuật toán xử lý (Bucket-level Accuracy), hệ thống đạt độ chính xác **87.88%** nhờ vào chính sách quyết định tối ưu (No-retrain Policy) kết hợp với luật fallback.

### HĐ: Tại sao các em lại sử dụng và nhấn mạnh chỉ số dự đoán Top-2 (Top-2 Hit Rate)? Chỉ số này có ý nghĩa gì trong thực tế?
*   **Gợi ý trả lời:** Trong hệ thống thích ứng SmartBinocular, bộ phân loại môi trường không hoạt động độc lập để gán nhãn ngữ nghĩa, mà đóng vai trò bộ định tuyến chuyển mạch thuật toán. Nhiều lớp môi trường có sự tương đồng cao về thị giác và sử dụng chung một thuật toán xử lý (ví dụ: `normal_night` và `night_clear` đều được ánh xạ vào Bucket A). Do đó, chỉ số Top-2 Hit Rate (đạt **90.91%** trên toàn bộ dữ liệu cảm biến và **96.97%** trên phân khúc đêm IMX) phản ánh chính xác khả năng hệ thống giữ được tín hiệu định tuyến hợp lý, hỗ trợ đưa ra các gợi ý chuyển mạch phụ trợ (secondary hints) hoặc kích hoạt luật fallback khi độ tự tin Top-1 suy giảm.

### HĐ: Tập dữ liệu BU-TIV dùng đánh giá MAD có thực sự đại diện cho cảm biến nhiệt MI48 của hệ thống không?
*   **Gợi ý trả lời:** BU-TIV là một tập dữ liệu ảnh nhiệt hồng ngoại tiêu chuẩn của thế giới, được chúng em sử dụng như một **tập dữ liệu thay thế kiểm thử ngoài (external surrogate benchmark)** để đánh giá thuật toán trong điều kiện có nhãn chuẩn (ground truth), điều mà tập dữ liệu nội bộ chưa đáp ứng được. Để mô phỏng sát nhất giới hạn vật lý của cảm biến nhiệt MI48, chúng em đã down-sample toàn bộ ảnh BU-TIV về đúng độ phân giải **80x62 pixels** và chuyển đổi bounding box XML sang mặt nạ nhị phân ở độ phân giải này. Tuy nhiên, chúng em hoàn toàn ghi nhận đây là dữ liệu surrogate và việc kiểm chứng trực tiếp trên luồng nhiệt bức xạ raw từ module cảm biến MI48 thực tế vẫn là một cổng nghiệm thu (validation gate) cần thực hiện trong tương lai.

### HĐ: Tại sao các em không tiến hành huấn luyện lại (retrain) mô hình Random Forest khi phát hiện hiện tượng trôi phân phối cảm biến?
*   **Gợi ý trả lời:** Việc huấn luyện lại mô hình học máy yêu cầu một lượng dữ liệu nhãn cảm biến thực tế đủ lớn và cân bằng giữa 9 lớp môi trường. Hiện tại, tập nhãn thủ công sau khi tinh lọc và duyệt thị giác chỉ có **132 khung hình** được giữ lại, và hầu hết các lớp thời tiết bất lợi (mưa, sương mù) hoặc chuyển giao (dawn/dusk) có số lượng mẫu cực kỳ ít (low-support). Việc huấn luyện lại trên tập dữ liệu nhỏ này chắc chắn sẽ dẫn đến hiện tượng quá khớp (overfitting) và mất đi các đặc trưng tổng quát hóa mà mô hình đã học từ tập dữ liệu lớn ban đầu. Do đó, phương án tối ưu nhất được chọn là xây dựng một **chính sách quyết định không huấn luyện lại (no-retrain policy)** ở lớp ứng dụng (phối hợp ngưỡng tin cậy, biên xác suất, gợi ý Top-2 và luật fallback), giúp nâng độ chính xác định tuyến bucket xử lý lên **87.88%** mà vẫn giữ nguyên mô hình phân loại gốc hoạt động ổn định.

### HĐ: MAD là một thuật toán phát hiện dị thường nhiệt, tại sao các chỉ số pixel F1-score (0.0829) và IoU (0.0294) của nó trên tập BU-TIV lại cực kỳ thấp như vậy?
*   **Gợi ý trả lời:** Các chỉ số đo đạc chồng lấn mức độ điểm ảnh (như pixel F1 và IoU) chịu ảnh hưởng cực kỳ lớn bởi độ phân giải của ảnh cảm biến. Khi chúng em down-sample ảnh về độ phân giải vật lý **80x62 pixels**, một vật thể dị thường (như một người đi bộ ở khoảng cách xa) chỉ chiếm diện tích từ **1 đến vài pixel** trên khung hình. Trong điều kiện kích thước đối tượng siêu nhỏ như vậy, chỉ cần lệch biên định vị đi 1 pixel, chỉ số IoU lập tức rơi về cận 0. Do đó, các chỉ số này chỉ đóng vai trò chẩn đoán chất lượng biên cục bộ. Đối với một bộ cảnh báo dị thường chạy ngầm có độ phân giải thấp, chỉ số quan trọng hơn là **Frame Precision (đạt 99.05%)** - tức là khi có cảnh báo xuất hiện trên màn hình HUD, người dùng hoàn toàn có thể tin cậy rằng thực sự có nguồn nhiệt dị thường trong vùng quan sát, giảm thiểu tối đa sự xao nhãng do báo động giả.

---

## 12. Checklist Chuẩn Bị Bảo Vệ Luận Văn

### Những số liệu cốt lõi bắt buộc phải nhớ
1.  **584 cặp ảnh:** Số lượng cặp khung hình NIR/Thermal hiển thị đồng bộ thu nhận thực tế từ thiết bị phục vụ đánh giá hợp nhất ảnh ngoại tuyến.
2.  **270 khung hình tĩnh & 1,620 dòng thử nghiệm:** Quy mô tập kiểm thử tĩnh phục vụ đánh giá phân phối bucket và kiểm chứng IQA ngoại tuyến.
3.  **14,094 bản ghi:** Quy mô tập dữ liệu huấn luyện và kiểm thử mô hình phân loại môi trường cơ sở (RGB-proxy).
4.  **132 khung hình nhãn thủ công:** Số lượng mẫu cảm biến thực tế được giữ lại để đánh giá trôi phân phối và hiệu năng bộ phân loại miền cảm biến.
5.  **87.88%:** Độ chính xác định tuyến bucket xử lý (Bucket-level accuracy) sau khi áp dụng chính sách giảm thiểu lỗi (No-retrain policy: $\tau=0.8, margin=0.05, hint=0.15$).
6.  **99.05%:** Frame precision của thuật toán MAD phát hiện dị thường nhiệt trên tập heldout BU-TIV 80x62.
7.  **16.9 FPS:** Tốc độ khung hình trung bình đo được khi chạy luồng xử lý thích ứng thực tế trên Raspberry Pi 4B.
8.  **28.2 phút:** Phiên chạy liên tục dài nhất ghi nhận độ ổn định hệ thống nhúng thời gian thực.
9.  **51 module pytest:** Quy mô tập kiểm thử tự động bảo vệ tính ổn định của mã nguồn hệ thống.

### Những nhận định nên nói (Claims to make)
*   Nhấn mạnh tính thực tiễn và tính khả thi của một **thiết bị nghiên cứu nguyên mẫu (research prototype)** hoạt động thích ứng trong điều kiện nhúng giới hạn tài nguyên.
*   Trình bày rõ ràng cơ chế định tuyến bucket (Bucket dispatch) giúp tối ưu hóa chất lượng ảnh và ngăn ngừa các lỗi nghiêm trọng của bộ lọc cố định (như việc chạy Dehaze ban đêm gây mất chi tiết tối).
*   Khẳng định giải pháp giảm thiểu lỗi trôi phân phối cảm biến không cần huấn luyện lại (No-retrain policy) là giải pháp thực tế, tối ưu hóa tài nguyên và có tính an toàn hệ thống cao.
*   Giới thiệu MAD là một **bộ chỉ thị cảnh báo sớm dị thường nhiệt nhẹ và tin cậy** với tiềm năng tích hợp chạy nền gọn nhẹ dựa trên hiệu năng máy trạm (0.1527 ms/khung hình).

### Những nhận định không nên nói (Claims to avoid)
*   *Tránh nói:* "Hệ thống đã được thử nghiệm và nghiệm thu hoàn toàn ngoài thực địa chiến đấu / thương mại." (Luôn định vị là nghiên cứu nguyên mẫu phòng thí nghiệm - research prototype).
*   *Tránh nói:* "Mô hình phân loại môi trường đã được tối ưu hóa hoàn toàn cho mọi điều kiện thời tiết." (Thừa nhận các lớp mưa, sương mù, chuyển giao vẫn đang thiếu dữ liệu hỗ trợ thực tế - low support).
*   *Tránh nói:* "Thuật toán MAD đã được tối ưu hóa để định vị chính xác tuyệt đối hình dáng biên đối tượng nhiệt." (Thừa nhận IoU và Pixel F1 thấp do giới hạn vật lý của độ phân giải cảm biến 80x62).

### Các giới hạn cần chủ động trình bày trước (Proactive Caveats)
*   Thừa nhận nhãn thực tế cảm biến là **nhãn thủ công được duyệt thị giác (manual labels)** chứ không phải nhãn chuẩn gold độc lập từ nhiều chuyên gia gán nhãn chéo.
*   Chủ động chỉ ra rằng tập BU-TIV là **dữ liệu surrogate bên ngoài** được resize về 80x62 để đánh giá thuật toán MAD, và dữ liệu thực tế từ luồng raw radiometric thermal của cảm biến MI48 vẫn đang là đích đến tiếp theo.
*   Xác nhận các thử nghiệm chính sách quyết định thích ứng (No-retrain policy) hiện tại đang dừng ở mức **mô phỏng trên log đặc trưng ngoại tuyến (offline feature-log simulation)** và cần được nghiệm thu trực tiếp trên luồng runtime thời gian thực của thiết bị nhúng.
