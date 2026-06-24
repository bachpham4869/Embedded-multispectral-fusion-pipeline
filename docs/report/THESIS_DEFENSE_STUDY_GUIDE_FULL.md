# SmartBinocular Thesis Defense Study Guide

## 0. Executive Summary

SmartBinocular là một research prototype dạng binocular-style dùng Raspberry Pi 4B để kết hợp hai nguồn quan sát bổ sung cho nhau: camera NIR độ phân giải cao và cảm biến nhiệt MI48 độ phân giải thấp. Mục tiêu không phải tạo một sản phẩm quân sự/commercial đã được chứng nhận, mà là chứng minh một kiến trúc nhúng chi phí thấp có thể thu nhận, xử lý, định tuyến thuật toán, hợp nhất ảnh, hiển thị HUD và ghi telemetry theo cách có thể kiểm chứng bằng artifact.

Vấn đề chính của hệ thống là quan sát trong đêm, low-light, fog/haze, glare/backlight và các tình huống có thermal anomaly. Một camera NIR như Sony IMX290 giữ được texture và spatial detail tốt hơn thermal, nhưng phụ thuộc vào ánh sáng phản xạ hoặc chiếu NIR. Thermal/LWIR từ MI48 không phụ thuộc ánh sáng, nhìn được hot foreground, nhưng chỉ 80x62 nên localization, IoU và detail rất hạn chế. SmartBinocular ghép hai hướng: NIR là kênh texture chính, thermal là kênh heat cue/foreground cue.

Pipeline chính gồm:

1. Capture NIR và thermal-display/heatmap-like frame.
2. Tạo `FrameCache`, resize/gray/statistics cho xử lý nhanh.
3. Trích xuất 12 handcrafted optical features.
4. Environment classification bằng Random Forest.
5. Hysteresis / fallback / top-2 policy để ổn định ENV state.
6. Bucket dispatch A-F cho NIR processing.
7. Thermal preprocessing: 3D-NR/EMA, background model/Kalman, heat map, mask/MAD-style anomaly indicator.
8. Fusion compositor: alpha blending hoặc foreground-mask overlay sau alignment/warp.
9. HUD/display và telemetry/session logging.

Các contribution chính:

- Low-cost dual-sensor embedded research prototype.
- Environment-aware adaptive processing pipeline.
- Six semantic processing buckets thay vì một filter cố định.
- Lightweight Random Forest classifier trên 12 handcrafted features.
- Duplicate-cluster-aware split để giảm data leakage trong offline ML benchmark.
- Sensor-domain/domain-shift audit và no-retrain RuleFallback mitigation.
- Q1 fusion evaluation trên 584 live-captured paired inputs, nhưng fusion output là `GENERATED_OFFLINE`.
- Q2 bucket/IQA evaluation trên 270 validation frames và 1,620 forced-bucket rows.
- Q3 MAD thermal anomaly indicator benchmark trên BU-TIV external surrogate resized 80x62.
- Partial Raspberry Pi 4B profiling với session telemetry.

Các kết quả nổi bật cần nhớ:

| Evidence area | Key result | Claim allowed | Caveat |
| --- | ---: | --- | --- |
| Dataset | 14,094 rows; 11,981 train / 2,113 test | Offline RGB-proxy reference dataset | Not live NIR/LWIR distribution |
| RF200 | Acc 0.8263; balanced acc 0.7463; macro-F1 0.7362 | Current offline baseline | macOS latency proxy, not RPi4 proof |
| RF100 | Acc 0.8230; balanced acc 0.7415; macro-F1 0.7325 | Embedded candidate | target-hardware feature+predict latency remains separate |
| Sensor retained | n=132; Top-1 0.7576; Top-2 0.9167; macro-F1 0.5818 | User-selected official sensor-domain evidence | manual/user-approved labels, not independent gold labels |
| No-retrain policy | bucket accuracy 87.88%; fallback 18.32% | Practical dispatch mitigation | offline feature-log simulation |
| Q1 fusion | 584 strict paired NIR/thermal-display inputs | Foreground-mask overlay improves contrast/edge proxies | offline-generated fusion, not runtime triple capture |
| Q2 bucket | 270 frames / 1,620 rows | Fixed filters can fail badly out of condition | still-image/forced-bucket evidence |
| MAD | precision 0.9905; recall 0.3772; IoU 0.0294 | Lightweight visual-warning indicator on surrogate thermal data | BU-TIV, not MI48 raw/radiometric field validation |
| RPi4 timing | 30 sessions; 16.9 FPS throughput-profile; longest 28.2 min | Partial target-hardware profiling | not full mode-matrix acceptance |

Điểm quan trọng nhất khi bảo vệ là không overclaim. Thesis có nhiều bằng chứng tốt, nhưng từng claim phải đứng đúng evidence tier:

- ML offline tốt không đồng nghĩa sensor-real 9-class classifier fully validated.
- Sensor-domain labels không phải gold labels.
- Fusion từ paired inputs là generated offline, không phải captured runtime fusion.
- Thermal-display/heatmap-like không phải raw radiometric thermal.
- BU-TIV MAD là surrogate benchmark, không phải MI48 field validation.
- Timing là partial profiling, không phải full deployment acceptance.

## 1. Thesis One-Minute Pitch

SmartBinocular giải quyết bài toán quan sát ban đêm và điều kiện xấu bằng một prototype nhúng chi phí thấp kết hợp NIR và thermal. NIR cung cấp texture và độ phân giải, thermal cung cấp heat cue không phụ thuộc ánh sáng. Thay vì áp dụng một thuật toán cố định cho mọi scene, hệ thống phân loại môi trường bằng Random Forest từ 12 handcrafted features, sau đó định tuyến frame qua sáu processing buckets như night enhancement, CLAHE, anti-glare, fog dehaze, rain temporal median và transition blend.

Evidence chính gồm: offline duplicate-cluster-aware ML benchmark với RF200 đạt 0.8263 accuracy và 0.7362 macro-F1; sensor-domain evaluation user-selected trên 132 retained frames với Top-2 0.9167; no-retrain policy đạt 87.88% processing-bucket accuracy; Q1 fusion trên 584 strict paired NIR/thermal-display inputs cho thấy foreground-mask overlay tăng foreground contrast và edge-density proxy; Q2 bucket/IQA chứng minh filter cố định có thể gây lỗi nặng, ví dụ dehaze phá night scene; Q3 MAD trên BU-TIV 80x62 đạt frame precision 0.9905 nhưng recall/IoU thấp.

Giới hạn phải nói rõ: đây là research prototype, không phải deployed product; fusion evidence là generated offline, sensor labels không phải independent gold, thermal là display/heatmap-like, MAD là surrogate, và timing mới là partial RPi4 profiling.

## 2. System Problem and Motivation

Quan sát ban đêm và thời tiết xấu khó vì mỗi sensor chỉ giải quyết một phần vấn đề. Low-light làm visible/NIR camera thiếu photon, noise tăng, contrast thấp. Fog/haze làm tán xạ ánh sáng, làm mất depth/edge. Glare/backlight làm vùng sáng bão hòa, kéo tone mapping sai và che mất foreground. Thermal anomaly lại không luôn rõ trên camera phản xạ vì NIR nhìn texture, không nhìn nhiệt độ.

NIR hữu ích vì camera như IMX290 có spatial resolution cao, có thể giữ cạnh, texture, đường, vật thể, background context. Nhưng NIR là reflected-light modality: trong zero illumination hoặc scene quá tối, ảnh có thể nghèo thông tin. Thermal/LWIR hữu ích vì đo phát xạ nhiệt, không cần ánh sáng môi trường, nhưng MI48 chỉ 80x62 nên không thể kỳ vọng segmentation sắc nét hoặc object localization chuẩn như camera độ phân giải cao.

Adaptive processing cần thiết vì thuật toán tốt trong điều kiện này có thể gây hại trong điều kiện khác. CLAHE tăng local contrast nhưng có thể khuếch đại noise. Dehaze có ích cho fog nhưng crush shadow trong night_clear. Anti-glare giúp vùng highlight nhưng có thể làm scene bình thường phẳng đi. Rain temporal median cần chuỗi frame, không thể đánh giá đầy đủ bằng still image.

Embedded/Raspberry Pi là constraint quan trọng vì hệ thống phải chạy trong budget CPU, memory, thermal và latency thấp. Các lựa chọn như Random Forest và handcrafted features không phải vì chúng hiện đại nhất, mà vì chúng giải thích được, nhẹ hơn deep model lớn, dễ log/debug và phù hợp với prototype nhúng.

Phạm vi defense nên nói: SmartBinocular là research prototype có evidence registry rõ ràng. Nó không phải military-grade/commercial certified product, chưa có full field validation, chưa có full mode-matrix acceptance và chưa có raw radiometric MI48 validation.

## 3. Research Questions and Objectives

| RQ/Object | Hỏi gì? | Đánh giá bằng gì? | Kết quả chính | Caveat |
| --- | --- | --- | --- | --- |
| Q1 Fusion utility | Fusion NIR + thermal có cải thiện tín hiệu hiển thị so với baseline không? | 584 strict paired NIR/thermal-display inputs; offline-generated fusion metrics | Foreground-mask overlay tăng foreground contrast delta +3.017641 và edge-density delta +0.012231 | Not runtime-captured fusion triples; not raw radiometric thermal |
| Q2 Adaptive bucket dispatch | Adaptive buckets có cần thiết hơn một pipeline cố định không? | 270 validation frames, 1,620 forced-bucket IQA rows | Bucket D trên night_clear crush 67.7%; adaptive routing tránh out-of-condition filter | Still-image offline; rain/transition temporal behavior not fully validated |
| Q3 Thermal MAD anomaly indicator | MAD thermal path hoạt động như detector hay indicator? | BU-TIV external surrogate resized 80x62, tune 160, heldout 3,322 | Frame precision 0.9905; runtime 0.1527 ms/frame; recall/IoU thấp | Surrogate, not MI48 real-field validation; indicator not classifier |
| O1 Fusion pipeline | Thiết kế/implement NIR-thermal fusion với evidence boundary | Source architecture, paired audit, fusion tables | Pipeline implemented; 584 live-captured paired inputs | Fusion output evaluated offline |
| O2 Adaptive processing | Xây sáu buckets và dispatch theo ENV class | `OPTICAL_BUCKET_DISPATCH`, Q2 IQA | Six branches A-F implemented; evidence for A-D, caveat for E-F | E/F need temporal/live validation |
| O3 Environment classifier/adaptive dispatch support | ML có đủ nhẹ và đủ tốt để hỗ trợ dispatch không? | 14,094-row offline benchmark; 132-frame official sensor eval; no-retrain policy | RF200 baseline; RF100 candidate; policy bucket accuracy 87.88% | Sensor domain still caveated; labels not gold |

## 4. Background / Theory

### 4.1 NIR Imaging

NIR là near-infrared, thường khoảng 0.7-1.0 micrometer với camera CMOS extended spectral response. NIR gần visible hơn thermal: nó vẫn là ánh sáng phản xạ, nên vật thể có texture và edge giống ảnh grayscale/visible. Trong night observation, NIR có lợi vì có thể dùng ambient NIR hoặc active IR illumination để tạo ảnh rõ hơn mắt người.

Giới hạn của NIR là nó không tự nhìn nhiệt. Nếu scene không có ánh sáng phản xạ, ảnh có thể rất tối. Nếu có glare/headlight, highlight saturation làm mất chi tiết. Nếu fog/haze, tán xạ làm contrast thấp. Vì vậy NIR cần enhancement theo điều kiện.

### 4.2 Thermal Imaging / LWIR

LWIR đo bức xạ nhiệt, thường 8-14 micrometer. Thermal hữu ích trong zero-light vì không cần ánh sáng phản xạ. Trong SmartBinocular, thermal modality trong fusion/eval hiện tại phải được gọi là `thermal-display` hoặc `display/heatmap-like`, không gọi là raw radiometric thermal nếu artifact không chứa raw radiometric arrays.

MI48 có độ phân giải 80x62. Đây là constraint vật lý lớn: một người hoặc hot object có thể chỉ chiếm vài pixel. Vì vậy pixel F1, mean IoU và object center recall dễ thấp dù frame-level warning precision cao. Hội đồng có thể hỏi vì sao IoU thấp; câu trả lời là ở 80x62, lệch 1-2 pixel đã làm overlap giảm mạnh, nên MAD phù hợp hơn như indicator/visual warning hơn là detector định vị biên chính xác.

### 4.3 Image Enhancement

CLAHE là Contrast Limited Adaptive Histogram Equalization. Nó chia ảnh thành tile, cân bằng histogram cục bộ và dùng clip limit để tránh khuếch đại noise quá mức. CLAHE tốt cho low-contrast NIR/fog-like scenes, nhưng nếu dùng sai clip hoặc sai scene có thể tạo artifact.

Gamma/tone mapping điều chỉnh quan hệ input-output intensity. Gamma nhỏ hơn 1 có thể nâng vùng tối; tone mapping highlight giảm vùng quá sáng để giữ chi tiết.

Highlight suppression/anti-glare dùng percentile/high-tail của luminance để phát hiện vùng sáng gần saturation, sau đó nén L-channel hoặc giới hạn display luminance. Mục tiêu là giảm cháy sáng, không phải làm ảnh "đẹp" tuyệt đối.

Dehazing/DCP Lite dựa trên Dark Channel Prior: trong ảnh haze-free, local patch thường có ít nhất một channel rất tối; haze làm dark channel sáng lên. DCP ước lượng transmission rồi phục hồi ảnh. Bản lite chạy ở độ phân giải giảm để tiết kiệm CPU. Trade-off: nếu áp dụng vào night_clear, thuật toán có thể crush shadow nặng.

Temporal median dùng nhiều frame để giảm streak/noise, phù hợp rain/wet/noisy sequence. Nó không thể được chứng minh đầy đủ bằng một still image.

3D-NR/EMA là temporal smoothing:

```text
T_t = alpha * I_t + (1 - alpha) * T_{t-1}
```

Trong đó `I_t` là frame hiện tại, `T_{t-1}` là estimate trước đó. Alpha lớn nghĩa là response nhanh hơn, smoothing ít hơn. Alpha nhỏ nghĩa là smoothing mạnh hơn nhưng lag nhiều hơn.

### 4.4 Fusion Theory

Alpha blending là phép kết hợp tuyến tính:

```text
F = (1 - alpha_T) * I_NIR + alpha_T * T_warped
```

`I_NIR` là ảnh NIR, `T_warped` là thermal frame đã warp/alignment sang không gian NIR, `alpha_T` là trọng số thermal. Alpha blending rẻ và ổn định, nhưng nếu thermal overlay phủ đều toàn ảnh thì có thể làm mất texture NIR.

Foreground-mask overlay dùng thermal mask để nhấn mạnh vùng foreground nóng thay vì pha thermal đều mọi pixel. Ý tưởng: giữ NIR texture ở background, chỉ đưa thermal cue vào vùng có khả năng là foreground/hot object. Q1 evidence cho thấy cách này tăng foreground contrast proxy và edge-density proxy so với alpha blend.

Homography/spatial alignment quan trọng vì hai sensor khác vị trí, field of view và resolution. Nếu alignment sai, thermal cue bị đặt lệch so với vật thể NIR. Thesis hiện có capture skew/cadence evidence, nhưng alignment RMSE chưa được fully validated, nên không claim spatial registration perfect.

### 4.5 Machine Learning Theory

Random Forest là ensemble của nhiều decision trees. Mỗi tree học từ bootstrap sample và feature splits; output là vote/probability aggregate. Ưu điểm trong thesis:

- chạy được trên feature vector nhỏ;
- ít cần tuning hơn deep net;
- xử lý non-linear thresholds tốt;
- dễ giải thích bằng feature importance, per-class metrics;
- phù hợp 12 handcrafted features và embedded constraint.

RF hyperparameters cần nhớ ở mức defense: `n_estimators` khác nhau giữa RF100 và RF200; `max_depth`, `min_samples_leaf`, `class_weight`, `n_jobs` là các knobs thường dùng để cân bằng accuracy/latency/model size. Nếu hội đồng hỏi chi tiết exact table, trả lời rằng RF200 current config là offline baseline và RF100 là lightweight candidate; exact hyperparameter table cần verify từ model bundle/report nếu slide yêu cầu.

MLP32, ExtraTrees và HistGradientBoosting được so sánh để chứng minh lựa chọn không tùy tiện. MLP32 rất nhanh trong proxy latency nhưng balanced accuracy và macro-F1 thấp hơn, nghĩa là yếu hơn ở class-balanced behavior. HistGradientBoosting có accuracy cao nhưng balanced accuracy/macro-F1 không tốt bằng RF200 theo tiêu chí chính. ExtraTrees cạnh tranh nhưng macro-F1 thấp hơn RF200.

Accuracy đo tỷ lệ dự đoán đúng tổng thể, dễ bị class imbalance che lấp. Balanced accuracy là trung bình recall theo class, giảm bias về class nhiều mẫu. Macro-F1 là F1 trung bình không trọng số theo class, quan trọng khi low-support classes như glare/backlight/transition dễ bị bỏ qua. Top-1 là class xác suất cao nhất; Top-2 kiểm tra label thật có nằm trong hai class đầu không, hữu ích khi hai class thị giác chồng lấn nhưng routing bucket vẫn tương tự.

Data leakage là khi train/test chia không độc lập, ví dụ gần trùng frame từ cùng video rơi vào cả hai split. Duplicate-cluster-aware split nhóm các ảnh trùng/gần trùng theo hash/split group rồi đưa cả cluster vào cùng một split. Đây là lý do offline benchmark đáng tin hơn split ngẫu nhiên đơn giản. Tuy vậy source overlap vẫn còn, nên không gọi là source-held-out benchmark.

Domain shift là khi phân phối feature train khác phân phối sensor runtime. Trong SmartBinocular, train domain là RGB-proxy optical data, còn sensor-domain gồm NIR/grayscale IMX và day sensor frames. KS drift gần 1.0 ở IMX night chứng minh model gặp input distribution rất khác training reference.

No-retrain mitigation là một policy layer không thay model: chỉ chấp nhận RF khi confidence/margin đủ; nếu không, fallback rule/hint/bucket grouping xử lý. Cách này thực tế vì 132 retained sensor labels quá nhỏ để retrain 9-class model đáng tin.

### 4.6 MAD Thermal Anomaly Indicator

MAD là Median Absolute Deviation:

```text
MAD = median(|x_i - median(x)|)
modified_z_i = 0.6745 * (x_i - median(x)) / MAD
```

MAD robust hơn mean/std khi có outlier. Trong thermal frame, hot object có thể xuất hiện như outlier so với background. Nhưng SmartBinocular nên gọi đây là anomaly indicator, không phải detector đầy đủ, vì nó không phân loại object, không có multi-object tracker và spatial masks ở 80x62 rất coarse.

Frame precision/recall/F1 đo đúng-sai ở mức có cảnh báo trong frame. Pixel F1/IoU đo overlap mask, rất khắt khe ở 80x62. Object recall@3 đo tâm object có gần ground truth trong 3 pixel không. Kết quả BU-TIV cho thấy precision cao, recall và localization còn hạn chế.

## 5. System Design

### 5.1 Hardware Architecture

Hardware chính:

- Raspberry Pi 4B làm compute/control node.
- NIR/IMX sensor qua CSI-2, độ phân giải cao hơn thermal.
- Thermal/MI48 qua SPI/GPIO, 80x62, khoảng 9 FPS.
- 5-inch HDMI LCD/HUD display.
- Housing/cable layout để đặt hai sensor cùng hướng nhìn.

Data flow: NIR frame và thermal frame đi vào capture threads/cache; NIR giữ vai trò texture stream; thermal đi qua preprocessing/heatmap/mask; ENV control plane quyết định bucket; display plane composite output ra HUD.

### 5.2 Software Architecture

Software architecture gồm các lớp:

- Capture pipeline: đọc frame từ NIR và MI48/thermal-display.
- FrameCache: chuẩn hóa ảnh nhỏ, grayscale, BGR small, thermal 80.
- Feature extraction: 12 optical features bắt buộc.
- Environment classification: RF top-1/top-2 + probability.
- Bucket dispatch: map ENV class sang A-F.
- NIR processing: HybridNIREnhancer, CLAHE, anti-glare, dehaze, rain median, transition blend.
- Thermal processing: AGC, 3D-NR/EMA, background/Kalman, foreground mask, anomaly indicator.
- Fusion compositor: resize/warp/alpha/foreground mask overlay.
- HUD/display: render frame, debug state, telemetry.
- Session logging: JSON/CSV metrics, stage timing, frame counts.

Quan trọng: ENV update và bucket dispatch có one-frame offset trong thesis. Bucket cho frame hiện tại dùng stable ENV từ frame trước, còn feature extraction/classification của frame hiện tại update ENV cho frame sau. Điều này tránh circular dependency giữa "ảnh đã xử lý" và "feature dùng để chọn processing".

### 5.3 Environment Classes and Semantic Names

| Semantic name | Internal label | Meaning | Support/claim note |
| --- | --- | --- | --- |
| Clear Night | `night_clear` | Đêm tối/clear, ít ambient light | Strong offline support, sensor-real caveated |
| General Night | `normal_night` | Đêm có ambient/street light | Stronger than low-support classes |
| NIR Night Sensor Scene | `nir_night` | Mono/NIR-dominant night scene | Useful but sensor-domain caveated |
| Daylight Scene | `normal_day` | Ban ngày bình thường | Day sensor overlap with glare/backlight |
| Fog / Haze Scene | `fog` | Visibility degraded by haze/fog | Offline support, live weather support limited |
| Rain / Wet Scene | `rain` | Rain/wet/noisy streak conditions | Needs live temporal validation |
| Glare / Direct Highlight | `glare` | Direct light/highlight saturation | Low support/provisional |
| Backlit Scene | `backlight` | Subject against bright background | Low support/provisional |
| Dawn / Dusk Transition | `transition` | Transitional lighting | Treat as runtime transient/blend candidate |

Không nên nói rằng mọi class đã được validated sensor-real. Day/backlight/glare overlap mạnh trên grayscale/NIR; transition là runtime state hơn là class độc lập mạnh; rain/fog cần thêm real sensor/weather support.

### 5.4 Processing Buckets and Semantic Names

| Bucket | Semantic bucket | Internal policy | Goal | Evidence | Caveat |
| --- | --- | --- | --- | --- | --- |
| A | Hybrid Night Enhancement | `night_hybrid_enhance` | Tăng detail/contrast đêm, giữ texture | Q2/paired forced offline evidence | Not full runtime bucket validation |
| B | Monochrome CLAHE Enhancement | `nir_mono_clahe` | CLAHE nhẹ cho NIR mono night | Offline/proxy evidence | live NIR validation caveated |
| C | Highlight Suppression / Anti-glare Tone Mapping | `highlight_tone_map` | Nén highlight, chống glare/backlight | Parameter/evidence docs | class overlap day/glare/backlight |
| D | Fog Dehaze Lite | `fog_dehaze_lite` | Khử haze/fog bằng DCP lite | Q2 shows use and danger | Dangerous in night_clear if misrouted |
| E | Rain Temporal Median | `rain_temporal_median` | Giảm streak/noise theo chuỗi frame | Design implemented | real rain temporal validation pending |
| F | Dawn / Dusk Transition Blend | `dawn_dusk_blend` | Làm mềm chuyển bucket | Design implemented | weak evidence; runtime transition candidate |

### 5.5 Class-to-Bucket Mapping

| Semantic class | Internal label | Semantic bucket | Legacy bucket | Caveat |
| --- | --- | --- | --- | --- |
| Clear Night | `night_clear` | Hybrid Night Enhancement | A | confirmed by `env_presets.py` / `nir_pipeline.py` |
| General Night | `normal_night` | Hybrid Night Enhancement | A | confirmed |
| NIR Night Sensor Scene | `nir_night` | Monochrome CLAHE Enhancement | B | confirmed; sensor-domain caveated |
| Daylight Scene | `normal_day` | Highlight Suppression / Anti-glare Tone Mapping | C | confirmed; day/glare/backlight overlap |
| Fog / Haze Scene | `fog` | Fog Dehaze Lite | D | confirmed; live fog support limited |
| Rain / Wet Scene | `rain` | Rain Temporal Median | E | confirmed; temporal validation pending |
| Glare / Direct Highlight | `glare` | Highlight Suppression / Anti-glare Tone Mapping | C | confirmed; low-support/provisional |
| Backlit Scene | `backlight` | Highlight Suppression / Anti-glare Tone Mapping | C | confirmed; low-support/provisional |
| Dawn / Dusk Transition | `transition` | Dawn / Dusk Transition Blend | F | confirmed; runtime transient caveat |

## 6. Implementation Details

### 6.1 Feature Extraction

12 core optical features:

1. `nir_mean_brightness`
2. `nir_std`
3. `nir_entropy`
4. `nir_p95`
5. `nir_glare_score`
6. `nir_sharpness`
7. `nir_dark_fraction`
8. `nir_saturation_mean`
9. `hour_of_day_sin`
10. `hour_of_day_cos`
11. `prev_env_class`
12. `nir_blue_mean_ema`

Handcrafted features được dùng vì nhẹ, explainable, dễ compute trên Pi và dễ audit domain shift. Feature v2/21 có trong repo như research direction, nhưng chưa migration production vì cần schema/model/version approval và sensor validation. Không nên nói 21-feature model đã thay production.

### 6.2 Environment Classifier

RF200 là offline accuracy baseline; RF100 là lightweight tree candidate vì metric gần RF200 nhưng model nhỏ hơn. Runtime inference wrapper load joblib bundle, validate feature set, predict class probability và trả top-1/top-2. Confidence và margin dùng cho decision policy.

Top-2 có vai trò quan trọng vì nhiều class nhập nhằng nhưng bucket giống hoặc gần giống. Ví dụ day/backlight/glare có visual overlap; night_clear/normal_night/nir_night cũng nằm trên night axis.

Hysteresis và fallback giảm flicker. Hysteresis yêu cầu trạng thái mới tồn tại đủ frames trước khi commit, tránh switch liên tục. Fallback dùng rule khi RF confidence/margin không đủ.

### 6.3 Image Processing Buckets

Hybrid Night Enhancement dùng dark/bright channel weighting, adaptive CLAHE, detail sharpening, color/tone correction. Mục tiêu là nâng low-light texture mà không làm noise/glare quá mức.

CLAHE bucket dùng một pass đơn giản hơn cho NIR mono night, với clip control để tránh over-amplification.

Anti-glare/highlight tone mapping phát hiện high percentile/saturation rồi nén highlight. Nó dùng cho `normal_day`, `glare`, `backlight`.

Fog Dehaze Lite là DCP ở resolution giảm, rồi upsample. Nó tiết kiệm CPU nhưng có risk shadow crush ở scene không phải fog.

Rain Temporal Median cần nhiều frame để median theo thời gian. Nếu chỉ có single-frame proxy thì không đủ để claim live rain performance.

Transition Blend alpha-blend giữa output bucket trước/raw hoặc A/C-style output để tránh discontinuity khi ENV class chuyển.

### 6.4 Thermal Processing

Thermal preprocessing gồm:

- percentile AGC/stretch cho display;
- thermal 3D-NR/EMA;
- background estimate bằng warmup/EMA hoặc Kalman-like update;
- residual heat map;
- foreground mask bằng threshold/morphology;
- MAD/modified-z anomaly indicator.

Thermal path hiện tại nên nói là thermal-display/heatmap-like trong fusion evidence. Raw radiometric MI48 validation chưa được claim.

### 6.5 Fusion Implementation

Fusion nhận:

- NIR frame đã xử lý;
- thermal-display frame/heatmap-like frame;
- homography/warp hoặc resize/alignment;
- `alpha_T` base overlay;
- foreground mask để chỉ nhấn thermal cue ở vùng foreground.

Q1 official evidence dùng live-captured paired inputs nhưng offline-generated fusion outputs. Vì vậy câu đúng là: "offline-generated fusion from live-captured paired NIR/thermal-display inputs improves selected proxy metrics." Câu sai là: "runtime compositor fusion quality has been validated."

### 6.6 Telemetry and Profiling

Telemetry/session logs ghi frame count, FPS, stage timings và mode/session metadata. Có 30 RPi4 sessions, trong đó có session dài 28.2 phút. Stage timing table cho throughput-profile dùng n=10 measurement windows. Timing chứng minh partial target-hardware profiling và optimization pressure points, không phải full deployment acceptance.

## 7. Evaluation and Benchmark Summary

| Eval | Goal | Dataset | Condition | Metrics | Result | Claim allowed | Caveat |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Dataset distribution | Biết support/imbalance | 14,094 rows | Offline RGB-proxy | counts, class/source distribution | 11,981 train / 2,113 test | Dataset is documented | not sensor-real |
| Data leakage | Giảm duplicate leakage | frozen train/test + cluster audit | offline | exact/SHA/group/duplicate checks | zero confirmed duplicate classes in audit where reported | split is stronger than random | not source-held-out |
| ML benchmark | chọn model | 12 features, same split | macOS proxy | acc, balanced acc, macro-F1, p95 | RF200 0.8263/0.7463/0.7362 | RF200 baseline, RF100 candidate | latency proxy |
| Sensor-domain | đo domain behavior | 132 retained official frames | manually reviewed/user-approved | Top-1, Top-2, macro-F1 | retained Top-2 0.9167 | preliminary sensor evidence | not gold labels |
| Domain shift | giải thích sensor gap | IMX/day linked features | feature drift | KS, delta, out-of-range | IMX KS up to 0.9799 | domain shift is real | feature diagnostic only |
| No-retrain policy | mitigate dispatch | 132 retained frames | offline feature-log simulation | exact, family, bucket, fallback | bucket 87.88% | practical dispatch layer | not live runtime validation |
| Q1 fusion | compare fusion algorithm | 584 strict paired inputs | generated offline | contrast, edge, entropy, skew | contrast delta +3.017641 | thermal-guided overlay promising | no runtime triples |
| Q2 bucket/IQA | prove adaptive need | 270 frames / 1,620 rows | offline forced buckets | contrast, saturation, crush | Bucket D night_clear crush 67.7% | fixed filters can fail | no full temporal validation |
| Q3 MAD | anomaly indicator behavior | BU-TIV 80x62 | external surrogate | precision/recall/F1/IoU/ms | precision 0.9905, IoU 0.0294 | lightweight indicator | not MI48 field validation |
| Runtime timing | profiling context | 30 sessions | RPi4 partial | FPS, p95 stage timing | 16.9 FPS throughput-profile | partial target profiling | not mode-matrix acceptance |

### 7.1 Dataset Distribution

Official distribution:

- Total: 14,094.
- Train: 11,981.
- Test: 2,113.
- Classes: 9.
- Imbalance ratio: 7.55 train / 7.58 test, confirmed in distribution artifacts with small rounding differences.

Per-class distribution:

| Semantic class | Internal label | Train | Test | Total |
| --- | --- | ---: | ---: | ---: |
| Clear Night | `night_clear` | 2,364 | 417 | 2,781 |
| General Night | `normal_night` | 1,995 | 352 | 2,347 |
| Daylight Scene | `normal_day` | 2,077 | 366 | 2,443 |
| Fog / Haze Scene | `fog` | 1,311 | 231 | 1,542 |
| Rain / Wet Scene | `rain` | 1,097 | 194 | 1,291 |
| Glare / Direct Highlight | `glare` | 340 | 60 | 400 |
| Backlit Scene | `backlight` | 313 | 55 | 368 |
| Dawn/Dusk Transition | `transition` | 438 | 77 | 515 |
| NIR Night Sensor Scene | `nir_night` | 2,046 | 361 | 2,407 |

Source distribution:

| Source | Total |
| --- | ---: |
| `offline_backlight` | 368 |
| `offline_darkface` | 2,781 |
| `offline_exdark_street` | 2,347 |
| `offline_glare_street` | 400 |
| `offline_gray_nir` | 2,407 |
| `offline_mwd` | 1,123 |
| `offline_weather11` | 2,068 |
| `offline_weather_time` | 2,600 |

Class imbalance matters because glare/backlight/transition have far fewer examples, so accuracy alone can look good while minority classes remain weak.

### 7.2 Data Leakage Control

Duplicate-cluster-aware split phát hiện exact/near duplicates hoặc split groups rồi giữ cả cluster trong train hoặc test. Mục tiêu là tránh model học cùng một frame/video gần trùng trong train rồi được test trên frame gần giống.

Audit trong repo nêu các lớp kiểm tra như exact row, SHA, split group, duplicate cluster, weak vector và dHash cross-split matches. Kết luận đúng khi bảo vệ: leakage control tốt hơn random split và hỗ trợ offline benchmark. Không nói rằng hoàn toàn source-held-out, vì source overlap vẫn tồn tại.

### 7.3 ML Offline Benchmark

All models use same 12 handcrafted optical features and same duplicate-cluster-aware split.

| Model | Accuracy | Balanced accuracy | Macro-F1 | P95 ms | Interpretation |
| --- | ---: | ---: | ---: | ---: | --- |
| RF100 | 0.8230 | 0.7415 | 0.7325 | 16.522 | Lightweight embedded candidate |
| RF200 | 0.8263 | 0.7463 | 0.7362 | 17.017 | Current offline baseline |
| ExtraTrees | 0.8008 | 0.7386 | 0.7149 | 16.285 | Competitive but weaker macro-F1 |
| HistGradientBoosting | 0.8282 | 0.7096 | 0.7238 | 31.402 | Highest acc among listed, weaker balanced behavior |
| MLP32 | 0.7941 | 0.6614 | 0.6758 | 0.112 | Fast proxy inference, but class-balanced weakness |

RF200 is chosen as offline baseline because balanced accuracy and macro-F1 are strongest among the practical candidates. RF100 is a deployment candidate because it stays close to RF200 while likely cheaper. Latency here is a macOS/offline proxy, not Raspberry Pi 4 proof.

### 7.4 Sensor-Domain Evaluation - Official Latest Version

Use these final user-selected official values in defense.

Label type: manually reviewed / user-approved visual labels, not independent multi-annotator gold labels.

Coverage:

- Total reviewed: 240.
- Retained: 132.
- Excluded: 108.
- Retained Day: 33.
- Retained IMX Night: 99.
- Ambiguous-boundary frames: 121.

Official sensor-domain metrics:

| View | n | Top-1 | Top-2 | Macro-F1 | Defense interpretation |
| --- | ---: | ---: | ---: | ---: | --- |
| Retained All | 132 | 0.7576 | 0.9167 | 0.5818 | Exact class is imperfect, Top-2 is strong enough to support dispatch reasoning |
| Retained Non-ambiguous | 119 | 0.7436 | 0.9231 | 0.5897 | Removing ambiguous boundaries keeps Top-2 high |
| High Confidence | 132 | 0.7674 | 0.9302 | 0.5938 | Strongest official positive sensor-domain view |
| Day subset | 33 | 0.5758 | 0.7576 | 0.6785 | Day/glare/backlight are visually overlapping |
| IMX Night subset | 99 | 0.8182 | 0.9697 | 0.3017 | High accuracy/Top-2, but macro-F1 low due narrow/imbalanced class distribution |

Vì sao Day khó: grayscale/NIR-like day frames làm `normal_day`, `backlight`, `glare` chồng lấn; các cue như sun direction, overexposure và background brightness không always separable bằng 12 features.

Vì sao IMX Night tốt hơn Day: nhiều IMX night frames rơi vào night-family rõ hơn. Tuy nhiên macro-F1 thấp vì support hẹp và prediction có thể tập trung vào một vài class.

Top-2 quan trọng vì system dùng classifier để dispatch bucket. Nếu true label nằm trong top-2 hoặc cùng family/bucket, decision policy có thể dùng secondary hint/fallback để vẫn chọn processing path hợp lý.

### 7.5 Domain Shift Analysis

Strongest IMX drift features:

| Feature | KS | Mean delta | Out-of-range |
| --- | ---: | ---: | ---: |
| `nir_dark_fraction` | 0.9799 | +0.6897 | 98.99% |
| `nir_mean_brightness` | 0.9725 | -98.8901 | 98.99% |
| `nir_p95` | 0.9451 | -141.4369 | 97.98% |
| `nir_entropy` | 0.9171 | -2.1596 | 95.96% |

Ý nghĩa: RGB-proxy train domain khác NIR/grayscale sensor domain. IMX night tối hơn, entropy thấp hơn, bright percentile thấp hơn. Do đó performance giảm không chỉ vì model yếu, mà vì input distribution khác rất mạnh. Đây là lý do no-retrain policy được ưu tiên hơn retrain nhỏ.

### 7.6 No-Retrain RuleFallback Mitigation

Final selected policy:

- `tau_accept = 0.8`
- `margin_accept = 0.05`
- `tau_hint = 0.15`
- Input: 132 retained sensor frames.
- Evaluation: offline feature-log simulation.
- Exact Top-1 Accuracy: 70.45%.
- Family-level Accuracy: 71.21%.
- Processing-bucket Accuracy: 87.88%.
- Fallback Rate: 18.32%.

Policy sweep context:

| Policy view | Bucket accuracy | Fallback | Meaning |
| --- | ---: | ---: | --- |
| RF only | 0.7803 | 0% | Baseline top-1 bucket routing |
| RF top-2 oracle | 0.9394 | diagnostic only | Upper bound, not deployable |
| Rule only | 0.7652 | 100% rule | Rules alone not enough |
| RF threshold tau 0.90 | 0.8939 | 40.91% | Higher bucket acc but too much fallback |
| Selected policy | 0.8788 | 18.32% | Balance of accuracy and fallback |

Vì sao không retrain: 132 labels quá nhỏ cho 9 classes, không độc lập gold, và có thể overfit/catastrophic forgetting. Bucket accuracy quan trọng hơn exact 9-class trong ứng dụng vì mục tiêu là route đúng processing bucket. Policy simulation chưa phải live runtime validation.

### 7.7 Q1 Fusion Evaluation

Evidence:

- 584 strict paired NIR/thermal-display frames.
- Inputs are live-captured paired frames.
- Fusion outputs are offline-generated from those inputs.
- Not runtime-captured fusion triples.

Metrics:

| Metric | Value |
| --- | ---: |
| Foreground contrast gain delta | +3.017641 |
| Foreground contrast win rate | 1.000000 |
| Edge-density delta | +0.012231 |
| Edge-density win rate | 0.958904 |
| NIR entropy mean | 5.709977 |
| NIR entropy delta | +1.028540 |
| NIR entropy win rate | 1.000000 |
| Thermal dynamic range mean | 157.376884 |
| Frame-skew p95 | 48.473850 ms |

Metric rows:

- fusion: 102,200.
- NIR: 56,064.
- thermal: 12,264.
- failure-case: 5,428.

Allowed claim: foreground-mask overlay improves selected proxy metrics on offline-generated fusion from live-captured strict paired inputs.

Caveats:

- no runtime fusion triples;
- no raw radiometric thermal;
- alignment RMSE not fully validated;
- edge density can reward artifacts, so interpret with visual/failure-case context.

### 7.8 Q2 Bucket/IQA Evaluation

Evidence:

- 270 independent validation frames.
- 1,620 forced-bucket rows.
- Still-image offline.

Key results:

| Finding | Value | Meaning |
| --- | ---: | --- |
| Fog Dehaze Lite / Bucket D on `night_clear` shadow crush | 67.7% | Misrouting dehaze into night scene is dangerous |
| Bucket D overall shadow crush | 18.29% | DCP is not safe as universal filter |
| CLAHE / Bucket B in fog log RMS contrast | 1.516 | CLAHE-like local contrast helps low-contrast fog-like frames |
| Bucket A saturation in fog | 7.84% | Hybrid night enhancer can over-saturate fog-like frames |

Caveat: this is offline still-image evidence. Real rain temporal median, dawn/dusk transition stability and full video hysteresis need live/sequence validation.
* **Log-RMS Contrast ($C_{\text{Log-RMS}}$):** A robust contrast metric tracking the standard deviation of pixel intensities in the logarithmic domain to map human/vision system responses.

$$C_{\text{Log-RMS}} = \sqrt{\frac{1}{W \cdot H} \sum_{x=1}^{W} \sum_{y=1}^{H} \big(\log I(x,y) - \overline{\log I}\big)^2}$$


### 7.9 Q3 MAD Anomaly Benchmark

BU-TIV output files exist in repo, so these metrics are confirmed from benchmark artifacts.

Benchmark design:

- External surrogate dataset: BU-TIV.
- 3,482 frames.
- Resized to 80x62 to match MI48 display resolution.
- Tune subset: 160 frames.
- Heldout/full-minus-tune: 3,322 frames.
- Selected config: per-frame percentile normalization, direct resize, z-threshold 2.5, min blob area 1.

Heldout metrics:

| Metric | Value |
| --- | ---: |
| Frame precision | 0.9905 |
| Frame recall | 0.3772 |
| Frame F1 | 0.5464 |
| Obj recall@3 | 0.1523 |
| Pixel F1 | 0.0829 |
| Mean IoU | 0.0294 |
| Mean runtime | 0.1527 ms/frame |

Interpretation: MAD is a high-precision visual-warning indicator on this external surrogate. It is not a full detector: recall is low, localization is coarse, and IoU is extremely low because objects occupy very few pixels at 80x62.

Caveat: BU-TIV is not MI48 real-field raw/radiometric validation. It uses mapping assumptions and resized thermal-like data.

### 7.10 Timing / Runtime Profiling

Evidence:

- 30 RPi4 sessions.
- 2 sessions >5 minutes.
- Longest session: 28.2 minutes.
- Throughput-profile mean FPS: 16.9, std 2.9.

Stage latency from Chapter 6 throughput-profile table:

| Stage | Mean ms | p50 ms | p95 ms |
| --- | ---: | ---: | ---: |
| Frame cache construction | 6.6 | 6.5 | 7.5 |
| NIR bucket dispatch | 15.9 | 14.3 | 20.9 |
| Thermal preprocessing | 1.5 | 1.4 | 1.7 |
| Fusion composite | 17.9 | 17.8 | 25.0 |

Allowed claim: partial target-hardware profiling shows practical throughput in tested profile and identifies optimization hotspots. Do not claim full mode-matrix acceptance or final deployment proof.

## 8. Main Technical Contributions

1. Low-cost dual-sensor research prototype on Raspberry Pi 4B.
2. Adaptive environment-aware processing pipeline with separated control plane and pixel data plane.
3. Semantic processing bucket design A-F that maps scene condition to algorithm choice.
4. Lightweight handcrafted-feature ML classifier with RF200/RF100 benchmark.
5. Duplicate-cluster-aware dataset evaluation to reduce duplicate leakage.
6. Sensor-domain domain-shift evaluation with manual/user-approved labels and label cleanup.
7. No-retrain fallback mitigation for practical bucket routing under domain shift.
8. Thermal-guided fusion evaluation on strict paired inputs with explicit `GENERATED_OFFLINE` caveat.
9. MAD thermal anomaly indicator surrogate benchmark.
10. RPi4 session profiling and telemetry evidence.

## 9. Strong Claims vs Caveated Claims vs Do-Not-Claim

| Strong / safe | Caveated | Do not claim |
| --- | --- | --- |
| Offline duplicate-cluster-aware ML benchmark exists and RF200 reaches 0.8263 accuracy / 0.7362 macro-F1 | Sensor-domain labels are manually reviewed/user-approved, not gold | Fully validated sensor-real 9-class classifier |
| Dataset distribution is documented at 14,094 rows | Sensor-domain results are preliminary and class support is narrow | Gold-label accuracy |
| RF100 is close to RF200 and is a lightweight candidate | RuleFallback is offline feature-log simulation | Domain shift solved |
| Official sensor retained Top-2 is 0.9167 | Fusion is generated offline from live-captured paired inputs | Full runtime fusion validation |
| Selected no-retrain policy reaches 87.88% bucket accuracy | Thermal is display/heatmap-like, not raw radiometric | Raw radiometric thermal validation |
| Q1 foreground-mask overlay improves selected proxy metrics | Edge density can reward artifacts | Perfect perceptual fusion quality |
| Q2 shows fixed filters can damage out-of-condition scenes | Rain/transition evidence remains weak | Full weather/field validation |
| BU-TIV MAD precision 0.9905 if framed as surrogate | BU-TIV is external surrogate, not MI48 field | Production detector performance |
| Partial RPi4 profiling exists | Timing is partial, not full mode matrix | Military-grade/commercial deployment |

## 10. Likely Committee Questions and Suggested Answers

1. Vì sao dùng Random Forest?
   - Vì feature vector nhỏ, tabular, non-linear và cần chạy nhẹ trên embedded hardware. RF giải thích được hơn deep model lớn và đạt macro-F1/balanced accuracy tốt nhất trong nhóm practical baseline.

2. Vì sao không dùng MLP dù nhanh?
   - MLP32 có proxy latency rất thấp nhưng balanced accuracy 0.6614 và macro-F1 0.6758 thấp hơn RF100/RF200. Với class imbalance, macro-F1 quan trọng hơn tốc độ đơn thuần.

3. Vì sao không retrain sau domain shift?
   - Sensor labels official chỉ 132 retained frames và không phải independent gold labels. Retrain 9-class trên tập nhỏ dễ overfit. Policy layer giữ production RF và thêm fallback an toàn hơn.

4. Vì sao 12 features đủ/chưa đủ?
   - Đủ để tạo offline baseline tốt và chạy nhẹ. Chưa đủ để claim sensor-real fully validated vì domain shift lớn; feature v2/21 vẫn là research direction.

5. Duplicate-cluster-aware split là gì?
   - Là split giữ ảnh trùng/gần trùng trong cùng train hoặc test cluster, tránh leakage khi frame từ cùng source/video xuất hiện ở cả hai bên.

6. Có xử lý data leakage chưa?
   - Có audit duplicate/cluster/hash theo artifact. Tuy vậy không gọi là source-held-out vì source overlap vẫn còn.

7. Dataset distribution thế nào?
   - 14,094 rows, 11,981 train, 2,113 test, 9 classes. Low-support classes gồm glare/backlight/transition.

8. Class imbalance xử lý sao?
   - Báo cáo balanced accuracy và macro-F1, không chỉ accuracy. Class support được đưa rõ trong distribution table và caveat.

9. Vì sao transition không bỏ?
   - Transition đại diện dawn/dusk hoặc boundary state cần runtime smoothing. Nhưng claim về transition phải caveated vì support và sensor validation còn yếu.

10. Vì sao Day subset khó?
    - Normal_day, backlight và glare chồng lấn mạnh trên ảnh sensor/grayscale; cùng một frame có thể vừa sáng, vừa backlit, vừa glare cue.

11. Vì sao IMX Night accuracy cao hơn Day?
    - IMX night frames thường rơi rõ vào night-family, ít class hơn. Nhưng macro-F1 thấp do distribution hẹp/imbalanced.

12. Top-2 có ý nghĩa gì?
    - Top-2 cho biết model giữ được tín hiệu thứ hai hợp lý khi top-1 nhập nhằng. Nó hữu ích cho secondary hint và fallback routing.

13. Bucket accuracy khác Top-1 thế nào?
    - Top-1 đòi đúng exact class. Bucket accuracy chỉ cần route đúng processing algorithm. Với adaptive processing, bucket accuracy gần mục tiêu vận hành hơn.

14. RuleFallback hoạt động ra sao?
    - Nếu RF confidence và margin đủ, chấp nhận RF. Nếu không, dùng rule fallback/top-2 hint/family grouping để chọn ENV/bucket.

15. RuleFallback đã chạy live chưa?
    - Chưa được claim như live validation. Kết quả 87.88% là offline feature-log simulation trên 132 retained frames.

16. Fusion khác alpha blending ở đâu?
    - Alpha blending pha thermal đều toàn ảnh. Foreground-mask overlay dùng thermal mask để nhấn vùng hot foreground, giữ texture NIR ở background.

17. Q1 fusion đã runtime validate chưa?
    - Chưa. Inputs là live-captured paired, nhưng fusion outputs trong eval là offline-generated. Runtime fusion triples là future validation gate.

18. Thermal-display khác raw radiometric thế nào?
    - Thermal-display là heatmap/8-bit visual representation. Raw radiometric là dữ liệu nhiệt gốc có giá trị đo nhiệt/ADC. Thesis không claim raw radiometric validation nếu artifact không có raw arrays.

19. Homography/alignment có vấn đề gì?
    - Alignment cần do sensor khác FOV/resolution/position. Có capture skew evidence, nhưng alignment RMSE chưa fully validated, nên không nói alignment hoàn hảo.

20. Q2 270 frames và 1,620 rows khác nhau ra sao?
    - 270 là số frame gốc; mỗi frame bị forced qua 6 buckets, tạo 270 x 6 = 1,620 metric rows.

21. Bucket D dehaze vì sao nguy hiểm ở night?
    - DCP giả định haze model; trong night_clear, nó có thể coi vùng tối là haze/low transmission và crush shadow, tạo 67.7% shadow crush.

22. MAD là detector hay indicator?
    - Indicator. Nó báo vùng thermal outlier/visual warning, không phân loại object và không chứng minh detector localization đầy đủ.

23. Vì sao MAD IoU thấp?
    - 80x62 làm object rất nhỏ. Lệch vài pixel làm IoU rơi mạnh. Precision frame-level mới là điểm mạnh hiện tại.

24. Timing 16.9 FPS có nghĩa gì?
    - Là throughput-profile average trong telemetry corpus, hỗ trợ partial target-hardware profiling. Không phải full mode-matrix acceptance.

25. Full mode-matrix acceptance là gì?
    - Test đủ NIR-only/thermal-only/fusion, profile khác nhau, ML on/off, long runs, p50/p95/p99, drops, throttle, resource/power.

26. Những gì sẽ làm nếu có thêm thời gian?
    - Capture runtime fusion triples, collect independent gold sensor labels, validate MI48 raw/radiometric MAD, run full mode matrix, collect live rain/fog/transition sequences.

27. Contribution lớn nhất của thesis là gì?
    - Một prototype nhúng dual-sensor có pipeline adaptive, evidence registry và evaluation boundary rõ ràng, tránh overclaim nhưng tạo baseline thực tế cho nghiên cứu tiếp theo.

28. Vì sao dùng handcrafted features thay vì CNN?
    - Vì mục tiêu là lightweight, explainable, auditable và chạy trên Pi. CNN có thể tốt hơn nếu có nhiều sensor labels và compute budget lớn hơn.

29. Vì sao macro-F1 được nhấn mạnh?
    - Macro-F1 không để class lớn áp đảo. Nó cho thấy model xử lý glare/backlight/transition tốt hay không.

30. Vì sao BU-TIV được dùng cho MAD?
    - Vì cần labeled thermal benchmark ngoài để có GT masks/boxes. Nó được resized 80x62 để mô phỏng limit MI48, nhưng vẫn là surrogate.

## 11. Slide Cheat Sheet

| Number | Meaning | Caveat |
| ---: | --- | --- |
| 14,094 | Total offline ML rows | RGB-proxy/offline |
| 11,981 / 2,113 | Train / test rows | duplicate-cluster-aware |
| 9 | Environment classes | not all sensor-real validated |
| 132 | Retained official sensor frames | manual/user-approved labels |
| 240 / 108 / 121 | Reviewed / excluded / ambiguous-boundary | conflict details in Section 14 |
| 0.8263 / 0.7463 / 0.7362 | RF200 accuracy / balanced / macro-F1 | offline baseline |
| 0.8230 / 0.7415 / 0.7325 | RF100 accuracy / balanced / macro-F1 | embedded candidate |
| 0.7674 / 0.9302 | High-confidence sensor Top-1 / Top-2 | official selected values |
| 87.88% | Selected policy bucket accuracy | offline feature-log simulation |
| 18.32% | Selected policy fallback rate | not live validation |
| 584 | Q1 strict paired frames | live-captured inputs |
| +3.017641 | Q1 foreground contrast delta | generated offline |
| 270 / 1,620 | Q2 frames / forced-bucket rows | still-image offline |
| 67.7% | Bucket D night_clear crush | demonstrates misrouting risk |
| 0.9905 | MAD frame precision | BU-TIV surrogate |
| 0.3772 | MAD frame recall | low recall |
| 0.0294 | MAD mean IoU | 80x62 localization limit |
| 16.9 FPS | RPi4 throughput-profile mean | partial profiling |
| 28.2 min | Longest session | not full acceptance |

## 12. Final Defense Narrative

Mở bài: "Luận văn của em tập trung vào bài toán quan sát ban đêm và điều kiện xấu bằng một prototype nhúng chi phí thấp. Thay vì dùng một sensor duy nhất, hệ thống kết hợp NIR để giữ texture và thermal để giữ heat cue."

Vấn đề: "NIR có độ phân giải cao nhưng phụ thuộc ánh sáng phản xạ; thermal không phụ thuộc ánh sáng nhưng MI48 chỉ 80x62. Vì vậy một modality không đủ. Ngoài ra các thuật toán enhancement không thể dùng cố định cho mọi scene: dehaze tốt cho fog nhưng có thể phá night scene."

Giải pháp: "SmartBinocular dùng Raspberry Pi 4B, camera IMX/NIR và MI48 thermal. Pipeline capture hai stream, trích xuất 12 optical features, phân loại ENV bằng Random Forest, ổn định bằng hysteresis/fallback, rồi route NIR qua sáu processing buckets. Thermal đi qua 3D-NR/background/foreground mask và được dùng trong fusion."

Thiết kế: "Hệ thống tách ENV control plane với pixel data plane. Bucket dispatch dùng stable ENV từ frame trước, tránh circular dependency. Six buckets gồm Hybrid Night, NIR CLAHE, Anti-glare, Fog Dehaze Lite, Rain Temporal Median và Transition Blend."

ML: "RF200 là offline baseline trên duplicate-cluster-aware split với accuracy 0.8263, balanced accuracy 0.7463 và macro-F1 0.7362. RF100 gần tương đương nên là embedded candidate. Trên sensor-domain official 132 frames, Top-2 retained đạt 0.9167 và high-confidence Top-2 đạt 0.9302, nhưng đây là manually reviewed labels, không phải gold labels. Domain shift được lượng hóa bằng KS drift rất cao trên IMX night."

Mitigation: "Thay vì retrain trên 132 labels, thesis chọn no-retrain policy. Với tau_accept 0.8, margin 0.05, tau_hint 0.15, policy đạt exact 70.45%, family 71.21%, processing-bucket 87.88%, fallback 18.32% trong offline feature-log simulation. Đây là cách thực tế hơn vì mục tiêu vận hành là route đúng bucket."

Fusion: "Q1 dùng 584 live-captured strict paired NIR/thermal-display inputs. Fusion outputs được generated offline. Foreground-mask overlay tăng foreground contrast delta +3.017641 và edge-density delta +0.012231 so với alpha blend. Em không claim runtime fusion validation vì chưa capture runtime fusion triples."

Eval: "Q2 dùng 270 validation frames và 1,620 forced-bucket rows. Kết quả quan trọng là Bucket D dehaze có thể crush 67.7% shadow trên night_clear nếu misrouted. Q3 MAD benchmark trên BU-TIV resized 80x62 đạt frame precision 0.9905 nhưng recall 0.3772 và IoU 0.0294, nên nó là indicator chứ không phải detector đầy đủ."

Caveats: "Các evidence tier được giữ riêng: offline ML, generated-offline fusion, surrogate MAD, partial RPi4 timing. Không có claim military-grade, raw radiometric validation, full weather field validation hay full mode-matrix acceptance."

Kết luận: "Contribution chính là một prototype nhúng dual-sensor có adaptive processing và evidence registry rõ ràng. Nó chứng minh hướng tiếp cận khả thi, đồng thời chỉ rõ validation gates tiếp theo: gold sensor labels, runtime fusion triples, MI48 raw validation và full mode-matrix profiling."

## 13. Final Checklist Before Defense

- [ ] Verify latest domain-shift artifacts and confirm official sensor numbers on slides.
- [ ] Verify BU-TIV MAD artifacts exist before saying metrics are confirmed.
- [ ] Verify RF hyperparameter/model table if asked beyond RF100/RF200 aggregate metrics.
- [ ] Verify class-to-bucket mapping against `src/smartbinocular/nir_pipeline.py`.
- [ ] Verify fusion captions say `GENERATED_OFFLINE` / generated offline.
- [ ] Verify no runtime fusion overclaim.
- [ ] Verify no "gold labels" wording for sensor-domain manual labels.
- [ ] Verify no "raw radiometric thermal" claim for heatmap/display evidence.
- [ ] Verify PDF final compile if TeX toolchain is available outside this environment.
- [ ] Verify slide numbers match report: 132, 87.88%, 584, 270/1,620, 0.9905, 16.9 FPS.
- [ ] Verify speaker notes include caveats before committee asks.

## 14. Source Files and Metric Verification

### 14.1 Source Files Read

Thesis LaTeX:

- `HK252-DATN-142/thesis.tex`
- `HK252-DATN-142/chapters/front/abstract.tex`
- `HK252-DATN-142/chapters/main/ch1-introduction.tex`
- `HK252-DATN-142/chapters/main/ch2-background.tex`
- `HK252-DATN-142/chapters/main/ch3-requirements.tex`
- `HK252-DATN-142/chapters/main/ch4-system-design.tex`
- `HK252-DATN-142/chapters/main/ch5-implementation.tex`
- `HK252-DATN-142/chapters/main/ch6-evaluation.tex`
- `HK252-DATN-142/chapters/main/ch7-conclusion.tex`
- `HK252-DATN-142/chapters/back/code-excerpts.tex`
- `HK252-DATN-142/chapters/back/design-docs.tex`
- `HK252-DATN-142/chapters/back/sweeps.tex`
- `HK252-DATN-142/chapters/back/test-suite.tex`
- `HK252-DATN-142/tables/ch6_evaluation/**/*.tex`
- `HK252-DATN-142/refs/example.bib`

Docs/evidence:

- `EVAL_BENCHMARK_DEFENSE_GUIDE.md`
- `EVAL_DOMAIN_SHIFT_MAD_WORK_REPORT.md`
- `THESIS_REFINEMENT_WORK_REPORT.md`
- `RECOVERY_TEST_EVAL_BENCHMARK_PLAN.md`
- `docs/PIPELINE_EVIDENCE_REGISTER.md`
- `docs/ml/PROCESSING_BUCKETS.md`
- `docs/ml/ML_EVIDENCE_READINESS.md`
- `docs/ml/MODEL_SELECTION_RATIONALE.md`
- `docs/tables/ml/dataset_distribution_reference.md`
- `docs/tables/ml/dataset_distribution_train.md`
- `docs/tables/ml/dataset_distribution_test.md`
- `docs/tables/ml/source_distribution.md`
- `docs/tables/ml/model_comparison_cluster_aware.md`
- `docs/tables/fusion/fusion_result_summary_for_report.md`
- `docs/tables/fusion/per_bucket_processing_eval.md`
- `docs/fusion/IMAGE_PROCESSING_EVALUATION.md`
- `docs/fusion/RUNTIME_TIMING_EVIDENCE.md`
- `docs/thesis_eval/timing_performance/tables/session_index.csv`
- `docs/thesis_eval/timing_performance/tables/stage_timing_by_mode.csv`

Review artifacts:

- `review_artifacts/label_v2_rule_fallback_domain_mitigation/LABEL_V2_RULE_FALLBACK_DOMAIN_MITIGATION_REPORT.md`
- `review_artifacts/label_v2_rule_fallback_domain_mitigation/metrics/env_eval_v1_vs_v2_summary.md`
- `review_artifacts/label_v2_rule_fallback_domain_mitigation/metrics/env_eval_v2_retained_all.md`
- `review_artifacts/label_v2_rule_fallback_domain_mitigation/metrics/env_eval_v2_high_confidence.md`
- `review_artifacts/label_v2_rule_fallback_domain_mitigation/mitigation/no_retrain_policy_selected_summary.md`
- `review_artifacts/label_v2_rule_fallback_domain_mitigation/mitigation/no_retrain_policy_selected.json`
- `review_artifacts/label_v2_rule_fallback_domain_mitigation/thesis_ready/thesis_ready_domain_shift_mitigation_table.md`
- `review_artifacts/butiv_mad_benchmark/mad_benchmark_metrics_selected.md`
- `review_artifacts/butiv_mad_benchmark/mad_benchmark_metrics_selected.json`
- `review_artifacts/butiv_mad_benchmark/thesis_ready_mad_benchmark_table.md`
- `review_artifacts/butiv_mad_benchmark/thesis_ready_mad_interpretation.md`

Architecture/source files read at architecture level:

- `src/smartbinocular/feature_schema.py`
- `src/smartbinocular/feature_extractor.py`
- `src/smartbinocular/ml_inference.py`
- `src/smartbinocular/nir_pipeline.py`
- `src/smartbinocular/thermal_pipeline.py`
- `src/smartbinocular/env_presets.py`
- `src/smartbinocular/main.py`
- `tools/evaluate_paired_fusion.py`
- `tools/domain_shift_report.py`
- `tools/relabel_sensor_datasets.py`
- `review_artifacts/eval_scripts/*.py`
- `review_artifacts/label_v2_rule_fallback_domain_mitigation/eval_scripts/*.py`

### 14.2 Metric Source and Status Table

| Metric / value used | Source path | Status | Note |
| --- | --- | --- | --- |
| Total dataset 14,094 | `docs/tables/ml/dataset_distribution_reference.md` | confirmed from file | row_count 14094 |
| Train/test 11,981 / 2,113 | `docs/tables/ml/model_comparison_cluster_aware.md` | confirmed from file | train_rows/test_rows |
| Per-class train/test counts | `docs/tables/ml/dataset_distribution_train.md`, `docs/tables/ml/dataset_distribution_test.md`, `docs/tables/ml/source_distribution.md` | confirmed from file | values reproduced in Section 7.1 |
| Source distribution totals | `docs/tables/ml/source_distribution.md` | confirmed from file | reference source counts |
| RF100 0.8230 / 0.7415 / 0.7325 / p95 16.522 | `docs/tables/ml/model_comparison_cluster_aware.md`; `HK252-DATN-142/tables/ch6_evaluation/ml_classifier/ml_offline_model_comparison_table.tex` | confirmed from file | macOS proxy latency |
| RF200 0.8263 / 0.7463 / 0.7362 / p95 17.017 | same as above | confirmed from file | current offline baseline |
| ExtraTrees, HistGradientBoosting, MLP32 rows | same as above | confirmed from file | same 12-feature split |
| Sensor Retained All n=132 Top-1 0.7576 Top-2 0.9167 Macro-F1 0.5818 | user final instruction; also present as selected row in `review_artifacts/label_v2_rule_fallback_domain_mitigation/metrics/env_eval_v1_vs_v2_summary.md` | user-selected official value | Do not show V1/V2 in main guide |
| Sensor High Confidence n=132 Top-1 0.7674 Top-2 0.9302 Macro-F1 0.5938 | user final instruction; source artifact has n=129 for that historical row | user-selected official value | User selected n=132 official; source has related row with n conflict |
| Day subset n=33 Top-1 0.5758 Top-2 0.7576 Macro-F1 0.6785 | user final instruction; `env_eval_v1_vs_v2_summary.md` row `day_sensor_template` | user-selected official value | Used only as final official metric |
| IMX Night n=99 Top-1 0.8182 Top-2 0.9697 Macro-F1 0.3017 | user final instruction; `env_eval_v1_vs_v2_summary.md` row `imx_paired_night` | user-selected official value | confirmed alignment with artifact |
| V2 retained_all 0.7500 / 0.9091 / 0.5678 | `review_artifacts/label_v2_rule_fallback_domain_mitigation/metrics/env_eval_v2_retained_all.md` | conflict found | Tracked only here; not used in main guide |
| Current LaTeX manual-label table n=203 | `HK252-DATN-142/tables/ch6_evaluation/ml_classifier/ml_manual_label_eval_table.tex` | conflict found | Older/stale relative to user official 132-frame instruction |
| No-retrain tau 0.8 margin 0.05 hint 0.15 | `review_artifacts/label_v2_rule_fallback_domain_mitigation/mitigation/no_retrain_policy_selected_summary.md` | confirmed from file | exact params |
| Exact 70.45%, family 71.21%, bucket 87.88%, fallback 18.32% | same summary + JSON | user-selected official value | file values 0.704545/0.712121/0.878788/0.183191 support rounded official |
| Older 85.7% bucket routing | `HK252-DATN-142/chapters/main/ch7-conclusion.tex` and current LaTeX domain-shift mitigation table | conflict found | Do not use as final official |
| Domain shift KS rows | `HK252-DATN-142/tables/ch6_evaluation/ml_classifier/domain_shift_features_table.tex` | confirmed from file | values copied in Section 7.5 |
| Q1 584 strict paired frames | `docs/tables/fusion/fusion_result_summary_for_report.md`; `HK252-DATN-142/tables/ch6_evaluation/fusion/fusion_result_summary_table.tex` | confirmed from file | generated offline caveat |
| Q1 foreground contrast delta +3.017641 | same as above | confirmed from file | Tier 3 proxy |
| Q1 edge-density delta +0.012231 / win 0.958904 | same as above | confirmed from file | edge artifacts caveat |
| NIR entropy mean 5.709977 delta +1.028540 win 1.000000 | same as above | confirmed from file | no-reference IQA |
| Thermal dynamic range mean 157.376884 | same as above | confirmed from file | display/heatmap-like |
| Frame-skew p95 48.473850 ms | same as above; `docs/fusion/RUNTIME_TIMING_EVIDENCE.md` | confirmed from file | pairing skew, not spatial alignment |
| Fusion/NIR/thermal/failure rows 102,200 / 56,064 / 12,264 / 5,428 | paired/fusion report artifacts | confirmed from file | source tables exist; detailed counts should be rechecked before slide export |
| Q2 270 frames / 1,620 rows | `HK252-DATN-142/chapters/main/ch6-evaluation.tex`; `EVAL_BENCHMARK_DEFENSE_GUIDE.md` | confirmed from file | still-image forced buckets |
| Bucket D night_clear crush 67.7% | `EVAL_BENCHMARK_DEFENSE_GUIDE.md`; Chapter 6 discussion | confirmed from file | key defense number |
| Bucket D overall crush 18.29% | `EVAL_BENCHMARK_DEFENSE_GUIDE.md` | confirmed from file | fixed filter risk |
| CLAHE fog log_rms 1.516 | `EVAL_BENCHMARK_DEFENSE_GUIDE.md` | confirmed from file | offline IQA |
| Bucket A fog saturation 7.84% | `EVAL_BENCHMARK_DEFENSE_GUIDE.md` | confirmed from file | offline IQA |
| BU-TIV files exist | `review_artifacts/butiv_mad_benchmark/mad_benchmark_metrics_selected.md`, `.json`, thesis-ready table | confirmed from file | existence checked before marking confirmed |
| BU-TIV heldout n=3,322 precision 0.9905 recall 0.3772 F1 0.5464 IoU 0.0294 runtime 0.1527 | `review_artifacts/butiv_mad_benchmark/mad_benchmark_metrics_selected.md` | confirmed from file | external surrogate only |
| RPi4 30 sessions / 28.2 min | `docs/thesis_eval/timing_performance/tables/session_index.csv`; Chapter 3/6 text | confirmed from file | partial profiling |
| Throughput-profile 16.9 FPS std 2.9 | `HK252-DATN-142/chapters/main/ch6-evaluation.tex`; `EVAL_BENCHMARK_DEFENSE_GUIDE.md` | confirmed from file | 10 throughput-profile sessions |
| Stage latencies 6.6/15.9/1.5/17.9 ms and p95 7.5/20.9/1.7/25.0 | `HK252-DATN-142/chapters/main/ch6-evaluation.tex` | confirmed from file | partial target-hardware profiling |
| RF hyperparameters beyond estimator count | model bundle / registry needed | needs verification | Do not present exact `max_depth`, `min_samples_leaf`, `class_weight`, `n_jobs` unless checked live |
| Compiled final PDF | `HK252-DATN-142/thesis.pdf` if present | needs verification | LaTeX source was sufficient for this guide; compile not run |

### 14.3 Conflicts Found

1. Sensor-domain metrics conflict:
   - User official values use 132 retained frames with Retained All Top-1 0.7576, Top-2 0.9167, Macro-F1 0.5818.
   - Repo v2 artifact also has 132 retained frames but reports Top-1 0.7500, Top-2 0.9091, Macro-F1 0.5678.
   - Current main guide follows user-selected official values; v2 details stay only in this conflict section.

2. High-confidence n conflict:
   - User official value says High Confidence n=132 with Top-1 0.7674, Top-2 0.9302, Macro-F1 0.5938.
   - `env_eval_v1_vs_v2_summary.md` historical high-confidence row shows those metrics with n=129.
   - Main guide follows user-selected n=132.

3. LaTeX stale sensor/domain-shift values:
   - Current LaTeX tables include n=203 manual-label eval and older 85.7% bucket-level mitigation.
   - User final plan requires 132 retained official sensor numbers and 87.88% bucket accuracy.
   - Main guide uses user-selected official values and marks LaTeX as stale/conflict for this guide.

4. Timing tables differ by profile:
   - Chapter 6 throughput-profile table gives 6.6/15.9/1.5/17.9 ms stage means and 16.9 FPS.
   - Other cross-session CSVs have different means because they aggregate broader modes/profiles.
   - Main guide uses Chapter 6 throughput-profile values and labels them partial profiling.

### 14.4 Needs Verification Before Final Slides

Các mục dưới đây được đánh dấu **CẦN VERIFY** trước khi đưa vào slide cuối hoặc bản PDF nộp hội đồng:

- CẦN VERIFY: whether the final thesis PDF has been recompiled after any LaTeX metric updates.
- CẦN VERIFY: RF hyperparameters from model bundle if slides include details beyond RF100/RF200 estimator count and aggregate metrics.
- CẦN VERIFY: whether any captured runtime fusion triples have been added after these artifacts; if not, keep `GENERATED_OFFLINE`.
- CẦN VERIFY: whether independent multi-annotator sensor labels exist; if not, keep "manual/user-approved labels, not gold".
- CẦN VERIFY: MI48 raw/radiometric thermal validation; if absent, keep BU-TIV MAD as surrogate only.
- CẦN VERIFY: full mode-matrix timing; if absent, keep partial profiling wording.
