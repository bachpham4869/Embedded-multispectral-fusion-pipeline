# SmartBinocular — Offline ML Pipeline Plan (Dataset-First, Optical-Primary)

**Version:** 3.1  
**Status:** PHASE 3B COMPLETE — Phase 3C (RPi wire + field collection) là bước tiếp theo  
**Thay thế:** Phase 3 trong `ML_HYBRID_SYSTEM_PLAN.md` (tasks 3.6/3.7/3.9 bị block do thiếu RPi field sessions)  
**Last Updated:** 2026-04-10

---

## Tiến độ thực tế (2026-04-10)

| Phase | Status | Artifact |
|-------|--------|----------|
| **3A — Infrastructure** | ✅ DONE | `feature_schema.py`, `feature_extractor.py`, `config.py` ML keys, tất cả tools |
| **3B-1 — Dataset pipeline** | ✅ DONE | 8 JSONL trong `logs/ml/`, tổng ~10MB |
| **3B-2 — Field collection** | ⚠️ PARTIAL | Classes "thiếu" đã được bổ sung bằng offline datasets (rgb channel), chưa có NIR thực địa |
| **3B-3 — Baseline training** | ✅ DONE | `models/baseline/rf_optical_only.joblib` (20 MB), cv_balanced_accuracy = **0.6729** (ngưỡng ≥ 0.60 ✅) |
| **3C — RPi wire + field NIR** | 🔲 TODO | Bước tiếp theo — xem §8 |

### Phân lớp train set hiện tại (`data/training/optical_only.jsonl`)

| ENV class | Số mẫu | Nguồn offline | Ghi chú |
|-----------|--------|--------------|---------|
| normal_day | 2000 (cap) | MWD, weather_time, weather11 | OK |
| normal_night | 1806 | weather_time (Night), ExDark | OK |
| indoor | 1792 | indoor_cvpr | OK |
| night_clear | 1602 | DarkFace | rgb channel (không phải NIR) |
| fog | 1542 | weather11, weather_time | OK |
| rain | 1291 | MWD, weather11, weather_time | OK |
| transition | 515 | MWD (Sunrise), weather_time (Dawn/Dusk) | Ít mẫu — cải thiện sau |
| glare | 400 | glare dataset | rgb channel — cần NIR field sau |
| backlight | 368 | backlight dataset | rgb channel — cần NIR field sau |
| **TOTAL** | **11316** | — | 4806 vào training split sau filter |

> **Ghi chú n_samples=4806**: `train_classifier.py` báo cáo training split (≈42% tổng sau filter + train/test split). 3 features `hour_of_day_sin/cos`, `prev_env_class` = importance 0.0 — bình thường với ảnh tĩnh, sẽ có tác dụng khi wire vào live RPi pipeline.

---

## 0. Bối cảnh & Nguyên tắc kiến trúc

### 0.1 Deployment Reality

SmartBinocular là **optical-first system**:

- **Camera optical (NIR/RGB) = luôn có** trên mọi cấu hình phần cứng
- **Thermal (MI48) = optional** — có thể vắng mặt trên một số cấu hình
- Hệ thống **KHÔNG phải multispectral** theo nghĩa "cần nhiều band để hoạt động"
- **Không được** thiết kế luồng ML/ENV sao cho "không có thermal thì không chạy" hoặc "nhãn chỉ tin được khi có thermal"

### 0.2 Dataset Priority

| Mức độ | Dataset | Mục đích |
|--------|---------|---------|
| **PRIMARY** | Image2Weather, Weather-Time (road) | Nguồn chính cho weather/ENV labels — optical RGB, có nhãn thật |
| **Supplement** | MWD (4-class), 11-class Weather | Bổ sung coverage cho lớp ít mẫu |
| **OPTIONAL** | LLVIP, KAIST | Chỉ khi cần context thermal alignment; KHÔNG làm backbone data cho ENV |

### 0.3 Phase 3 gốc bị block ở đâu

Tasks `[RPi]` trong Phase 3 gốc chưa thực hiện được:
- **3.6** — Field sessions (night, fog, glare, indoor, ...) → blocked
- **3.7** — Label sessions → blocked vì 3.6 chưa có
- **3.9** — Target ≥2700 labeled samples → blocked

### 0.4 Constraints cứng (không thương lượng)

- **[C1]** Optical pipeline luôn hoạt động độc lập, không phụ thuộc thermal
- **[C2]** `optical_only` model = ứng viên production chính trên mọi hardware
- **[C3]** `with_thermal` model = optional enhancement — phải graceful degrade về optical_only
- **[C4]** Không phá runtime pipeline RPi đang hoạt động
- **[C5]** Không đổi JSONL log format — chỉ add optional fields
- **[C6]** Không deploy model từ dataset-only lên production RPi chưa qua RPi validation
- **[C7]** Không giả lập temporal features từ ảnh tĩnh
- **[C8]** Không impute zero cho features thiếu — dùng `None`, filter cứng khi train
- **[C9]** Không coi weak_label là ground truth **khi dataset đã có nhãn gốc**
- **[C10]** Không normalize chung nhiều nir_channel — scaler per-group

---

## 1. Kiến trúc Tổng thể

### 1.1 Module Map

```
src/smartbinocular/
├── feature_schema.py     [NEW — Step 1]    Feature sets, FeatureRecord dataclass
├── feature_extractor.py  [NEW — Step 4]    Pure optical extraction, không import hardware
├── utils.py              [MODIFY — Step 2]  Thêm MLLogger
├── config.py             [MODIFY — Step 3]  Thêm ML config keys
├── main.py               [MODIFY — Step 11, LAST]  Wire FeatureExtractor + MLLogger
│
│   ─── KHÔNG ĐỤNG VÀO ────────────────────────────────────────────
├── hardware.py           ← hardware drivers
├── thermal_pipeline.py   ← import READ-ONLY bởi feature_extractor (optional path)
├── nir_pipeline.py       ← import READ-ONLY bởi feature_extractor
├── display_pipeline.py   ← không liên quan
├── motion.py             ← import READ-ONLY bởi feature_extractor
└── env_presets.py        ← không liên quan ở phase này

tools/
├── offline_pipeline.py   [NEW — Step 6]   Dataset runner chính
├── validate_schema.py    [NEW — Step 5]   JSONL schema validator
├── label_session.py      [NEW — Step 7]   Manual annotation
├── check_features.py     [NEW — Step 8]   Distribution analysis
└── mix_datasets.py       [NEW — Step 9]   Merge + filter JSONL

models/
├── train_classifier.py   [NEW — Step 10]  Training pipeline
├── baseline/             [DIR]  Output Phase 3B — KHÔNG deploy tự động
└── production/           [DIR]  Output Phase 3C+ — sau RPi validation

data/
├── weather/              [DIR]  Image2Weather, Weather-Time, MWD, 11-class
├── thermal_optional/     [DIR]  LLVIP, KAIST (chỉ khi cần thermal path)
└── training/             [DIR]  Pre-processed JSONL datasets cho training
```

### 1.2 Data Flow Diagram

```
══════════════════════════════════════════════════════════════════════
  OFFLINE PIPELINE — OPTICAL PATH (PRIMARY)
══════════════════════════════════════════════════════════════════════

Image2Weather / Weather-Time / MWD / 11-class
(RGB images + GROUND TRUTH weather labels)
    │
    ▼
WeatherDatasetSource.iter_frames()
    yields: (rgb_bgr, label_str, timestamp)   ← label từ dataset gốc, không heuristic
    │
    ▼
build_frame_cache(rgb_bgr, thermal_raw=None, ts)
    → FrameCache  (thermal_80 = dummy zeros, nir_channel="rgb")
    │
    ▼
FeatureExtractor.extract(cache, nir_channel="rgb", thermal_channel="none")
    → FeatureRecord
        - CORE (optical): filled
        - THERMAL: tất cả None  ← optical path không cần thermal
        - MOTION/TEMPORAL: filled nếu video, None nếu still
        - RUNTIME_ONLY: None
        - label: từ dataset gốc (KHÔNG heuristic)
    │
    ▼
MLLogger.log(record)
    → logs/ml/offline_weather_*.jsonl

══════════════════════════════════════════════════════════════════════
  OFFLINE PIPELINE — THERMAL OPTIONAL PATH (SUPPLEMENT)
══════════════════════════════════════════════════════════════════════

LLVIP (NIR) / KAIST (RGB+LWIR)
    │  [Chỉ khi cần bổ sung ngữ cảnh thermal hoặc NIR night scenes]
    ▼
LLVIPSource / KAISTSource
    yields: (nir_or_rgb_bgr, thermal_raw_or_None, timestamp)
    │
    ▼
FeatureExtractor.extract(..., nir_channel=..., thermal_channel=...)
    → FeatureRecord
        - CORE: filled
        - THERMAL: filled nếu KAIST (lwir), None nếu LLVIP
        - label: weak_label từ heuristic (vì LLVIP/KAIST không có ENV labels)
    │
    ▼
MLLogger.log(record)
    → logs/ml/offline_thermal_*.jsonl

══════════════════════════════════════════════════════════════════════
  FIELD COLLECTION (NIR thực địa — Phase 3B-2)
══════════════════════════════════════════════════════════════════════

RPi IMX290 NIR + manual ENV annotation (glare, indoor, backlight, night_clear)
    │  [Classes KHÔNG có trong weather datasets]
    ▼
MLLogger.log() → logs/ml/field_*.jsonl
label_session.py → manual ground truth

══════════════════════════════════════════════════════════════════════
  TRAINING
══════════════════════════════════════════════════════════════════════

tools/mix_datasets.py
    offline_weather + field_nir → data/training/optical_only.jsonl
    offline_thermal + field_nir → data/training/optical_thermal.jsonl (optional)
    │
    ▼
models/train_classifier.py
    ├── optical_only model  → models/baseline/rf_optical_only.joblib
    └── with_thermal model  → models/baseline/rf_with_thermal.joblib  (ablation)

══════════════════════════════════════════════════════════════════════
  RUNTIME (RPi) — KHÔNG THAY ĐỔI
══════════════════════════════════════════════════════════════════════

Picamera2 (IMX290 NIR) [luôn có] + MI48 (thermal) [optional]
    │
    ▼
FeatureExtractor.extract(cache, source="rpi")
    → FeatureRecord (thermal fields filled nếu MI48 available, else None)
    │
    ▼
EnvPresetController → optical_only model (luôn chạy)
                    + with_thermal model (chỉ khi MI48 available) [enhancement]
```

---

## 2. Modality Taxonomy

### 2.1 Trường modality trong FeatureRecord

| Trường | Giá trị | Ý nghĩa |
|--------|---------|---------|
| `nir_channel` | `"nir"` | Near-infrared ~700–1000 nm (IMX290, LLVIP IR) |
| `nir_channel` | `"rgb"` | Visible RGB (weather datasets, KAIST color) |
| `thermal_channel` | `"lwir"` | LWIR thermal 8–14 μm (MI48, KAIST thermal) |
| `thermal_channel` | `"none"` | Không có thermal — **đây là trạng thái mặc định bình thường** |

### 2.2 Dataset → Modality Mapping

| Dataset | `nir_channel` | `thermal_channel` | Label quality | Role |
|---------|--------------|------------------|--------------|------|
| Image2Weather | `"rgb"` | `"none"` | Ground truth (weather class) | **PRIMARY** |
| Weather-Time (road) | `"rgb"` | `"none"` | Ground truth (weather + time-of-day) | **PRIMARY** |
| MWD (4-class) | `"rgb"` | `"none"` | Ground truth (weather class) | Supplement |
| 11-class Weather | `"rgb"` | `"none"` | Ground truth (weather class) | Supplement |
| LLVIP (IR images) | `"nir"` | `"none"` | Weak (heuristic, vì không có ENV label) | Optional |
| KAIST (color+LWIR) | `"rgb"` | `"lwir"` | Weak (heuristic, vì không có ENV label) | Optional |
| RPi field collection | `"nir"` | `"lwir"` hoặc `"none"` | Ground truth (manual annotation) | Production data |

### 2.3 Quy tắc labeling theo dataset role

| Role | Labeling strategy | Lý do |
|------|------------------|-------|
| PRIMARY (Image2Weather, Weather-Time) | Dùng nhãn gốc của dataset trực tiếp | Dataset đã có ground truth — weak heuristic chỉ làm giảm chất lượng |
| Supplement (MWD, 11-class) | Dùng nhãn gốc; map sang ENV taxonomy | Nhãn gốc đã đủ tin cậy |
| Optional (LLVIP, KAIST) | Weak label từ heuristic (vì không có ENV label) | Không có lựa chọn khác; ghi rõ `label_confidence` |
| Field collection (RPi) | Manual annotation qua label_session.py | Ground truth tốt nhất |

---

## 3. ENV Taxonomy & Weather Label Mapping

### 3.1 ENV Classes của SmartBinocular

```python
ENV_CLASSES = [
    "night_clear",    # Đêm tối, quang đãng — đặc thù optical night vision
    "normal_night",   # Đêm có ambient light (đường phố, đô thị)
    "normal_day",     # Ban ngày điều kiện thường
    "fog",            # Sương mù
    "rain",           # Mưa
    "glare",          # Chói sáng (đèn pha đối diện, mặt trời trực tiếp)
    "backlight",      # Ngược sáng (nguồn sáng phía sau subject)
    "indoor",         # Trong nhà
    "transition",     # Chuyển tiếp (hoàng hôn, bình minh)
]
```

### 3.2 Weather Dataset Labels → ENV Classes

**Quy tắc mapping chung:**
- Mapping chỉ dùng khi dataset label **đủ rõ ràng** để suy ra ENV class
- Mapping phải được ghi rõ trong `tools/label_mapping.yaml` để reproducible
- Khi 1 weather label map sang nhiều ENV class có thể: chọn lớp phổ biến hơn hoặc skip

#### Image2Weather & Weather-Time (road) — PRIMARY

| Dataset label | ENV class | Confidence | Điều kiện / Ghi chú |
|--------------|-----------|------------|---------------------|
| Sunny | `normal_day` | 0.85 | Ban ngày, nắng |
| Cloudy | `normal_day` | 0.75 | Vẫn là điều kiện day thường |
| Rainy | `rain` | 0.90 | Mưa rõ ràng |
| Foggy | `fog` | 0.90 | Sương mù rõ ràng |
| Snowy | *(skip)* | — | Không có ENV class tương ứng; có thể map `rain` nếu cần volume nhưng ghi rõ |
| **time-of-day: Night** (Weather-Time) | `normal_night` | 0.80 | Đêm đô thị, có ambient light |
| **time-of-day: Dawn / Dusk** | `transition` | 0.75 | Hoàng hôn / bình minh |
| **time-of-day: Morning / Afternoon** | `normal_day` | 0.85 | Kết hợp với weather label |
| Night + no artificial light context | `night_clear` | 0.65 | Cần filter thêm; ưu tiên field collection |

#### MWD (4-class)

| Dataset label | ENV class | Confidence |
|--------------|-----------|------------|
| Shine | `normal_day` | 0.85 |
| Cloudy | `normal_day` | 0.75 |
| Rain | `rain` | 0.90 |
| Sunrise | `transition` | 0.80 |

#### 11-class Weather (supplement)

| Dataset label | ENV class | Confidence | Ghi chú |
|--------------|-----------|------------|---------|
| Fog / Smog | `fog` | 0.85 | Map cả hai vào fog |
| Rain | `rain` | 0.90 | |
| Sandstorm | `fog` | 0.60 | Visibility degradation tương tự |
| Lightning | *(skip hoặc `rain`)* | 0.50 | Ambiguous; bỏ nếu không đủ samples |
| Dew, Frost, Glaze, Hail, Rime, Rainbow, Snow | *(skip)* | — | Lớp hiếm / không match ENV taxonomy |

### 3.3 Classes KHÔNG cover bởi weather datasets → cần field collection

| ENV class | Lý do không có trong weather datasets | Nguồn dữ liệu |
|-----------|--------------------------------------|--------------|
| `glare` | Đặc thù optical near-field (đèn pha, nguồn sáng điểm) | NIR field collection + RPi |
| `backlight` | Ngược sáng camera-specific | NIR field collection + RPi |
| `indoor` | Không có trong weather datasets (by definition) | NIR field collection + RPi |
| `night_clear` | Night ở weather datasets thường là đô thị (ambient) | NIR field collection + RPi |

**Kết luận:** Phase 3B cần 2 nguồn bổ sung nhau:
1. **Weather datasets** → fog, rain, normal_day, normal_night, transition (có nhãn thật)
2. **NIR field collection** → glare, backlight, indoor, night_clear (manual annotation)

### 3.4 Label Mapping File

Mapping trên được lưu tại `tools/label_mapping.yaml` và import bởi `offline_pipeline.py`:

```yaml
# tools/label_mapping.yaml
# Dataset label → ENV class mapping
# confidence: ngưỡng tối thiểu để record được dùng trong training

image2weather:
  Sunny:  {env: normal_day,  confidence: 0.85}
  Cloudy: {env: normal_day,  confidence: 0.75}
  Rainy:  {env: rain,        confidence: 0.90}
  Foggy:  {env: fog,         confidence: 0.90}
  Snowy:  {env: null}  # skip

weather_time:
  weather:
    Sunny:  {env: normal_day,  confidence: 0.85}
    Cloudy: {env: normal_day,  confidence: 0.75}
    Rainy:  {env: rain,        confidence: 0.90}
    Foggy:  {env: fog,         confidence: 0.90}
    Snowy:  {env: null}
  time_of_day:
    Night:     {env: normal_night, confidence: 0.80}
    Dawn:      {env: transition,   confidence: 0.75}
    Dusk:      {env: transition,   confidence: 0.75}
    Morning:   {env: normal_day,   confidence: 0.85}
    Afternoon: {env: normal_day,   confidence: 0.85}

mwd:
  Shine:   {env: normal_day,  confidence: 0.85}
  Cloudy:  {env: normal_day,  confidence: 0.75}
  Rain:    {env: rain,        confidence: 0.90}
  Sunrise: {env: transition,  confidence: 0.80}

weather_11class:
  Fog:       {env: fog,  confidence: 0.85}
  Fog/Smog:  {env: fog,  confidence: 0.85}
  Rain:      {env: rain, confidence: 0.90}
  Sandstorm: {env: fog,  confidence: 0.60}
  # Các lớp còn lại: null (skip)
```

---

## 4. Feature Schema (FINAL)

### 4.1 Feature Set Constants

```python
# src/smartbinocular/feature_schema.py

# ── Atomic sets ───────────────────────────────────────────────────────────────

FEATURE_SET_CORE = [
    # Optical image statistics — tính từ NIR hoặc RGB, luôn available
    "nir_mean_brightness",       # 1
    "nir_std",                   # 2
    "nir_entropy",               # 3
    "nir_p95",                   # 4
    "nir_glare_score",           # 5
    "nir_sharpness",             # 6
    "nir_dark_fraction",         # 7
    "nir_saturation_mean",       # 8  mean HSV S trên BGR nhỏ [0,255]
    # Context
    "hour_of_day_sin",           # 9
    "hour_of_day_cos",           # 10
    "prev_env_class",            # 11
]  # 11 features — bắt buộc, không bao giờ None

FEATURE_SET_THERMAL = [
    # Thermal enhancement — chỉ khi MI48 available (OPTIONAL)
    "thm_mean",                  # 8
    "thm_std",                   # 9
    "thm_max",                   # 10
    "thm_p95_p05_delta",         # 11
    "thm_fg_fraction",           # 12
    "thm_anomaly_score",         # 13
]  # 6 features — None nếu thermal_channel = "none"

FEATURE_SET_MOTION = [
    # Motion — cần video sequence
    "motion_magnitude",          # 14
    "motion_jerk",               # 15
]  # 2 features — None nếu không phải video

FEATURE_SET_TEMPORAL = [
    # Temporal deltas — cần ≥10 frame window
    "nir_brightness_delta_10f",  # 18
    "thm_mean_delta_10f",        # 19
]  # 2 features — None nếu không có window

FEATURE_SET_RUNTIME_ONLY = [
    # RPi hardware state — không bao giờ train với features này
    "skew_ms",                   # 16
    "fusion_alpha",              # 17
]  # 2 features — None khi offline

# ── Composite sets (dùng trong training) ──────────────────────────────────────

FEATURE_SET_OPTICAL_ONLY = FEATURE_SET_CORE
# 11 features. Nguồn: mọi RGB/NIR image.
# → PRODUCTION PRIMARY: chạy trên mọi hardware config.

FEATURE_SET_OPTICAL_THERMAL = FEATURE_SET_CORE + FEATURE_SET_THERMAL
# 17 features. Nguồn: RPi NIR+LWIR, hoặc KAIST (rgb+lwir, không lý tưởng).
# → PRODUCTION ENHANCEMENT: chỉ khi MI48 available; falls back về OPTICAL_ONLY.

FEATURE_SET_OPTICAL_MOTION = FEATURE_SET_CORE + FEATURE_SET_MOTION + FEATURE_SET_TEMPORAL
# 15 features. Nguồn: video sequences.

FEATURE_SET_FULL_OFFLINE = (
    FEATURE_SET_CORE + FEATURE_SET_THERMAL +
    FEATURE_SET_MOTION + FEATURE_SET_TEMPORAL
)
# 21 features. Nguồn: thermal video sequences (KAIST) hoặc RPi.

FEATURE_SET_RUNTIME_FULL = FEATURE_SET_FULL_OFFLINE + FEATURE_SET_RUNTIME_ONLY
# 23 features. RPi only.
```

### 4.2 FeatureRecord Dataclass

```python
@dataclasses.dataclass
class FeatureRecord:
    # ── CORE OPTICAL — bắt buộc, không bao giờ None ──────────────────────────
    nir_mean_brightness: float
    nir_std: float
    nir_entropy: float
    nir_p95: float
    nir_glare_score: float
    nir_sharpness: float
    nir_dark_fraction: float
    nir_saturation_mean: float
    hour_of_day_sin: float
    hour_of_day_cos: float
    prev_env_class: int                    # 0 = unknown

    # ── THERMAL OPTIONAL — None nếu thermal_channel = "none" ─────────────────
    thm_mean: Optional[float] = None
    thm_std: Optional[float] = None
    thm_max: Optional[float] = None
    thm_p95_p05_delta: Optional[float] = None
    thm_fg_fraction: Optional[float] = None
    thm_anomaly_score: Optional[float] = None

    # ── MOTION — None nếu không phải video sequence ───────────────────────────
    motion_magnitude: Optional[float] = None
    motion_jerk: Optional[float] = None

    # ── TEMPORAL — None nếu không có ≥10 frame window ────────────────────────
    nir_brightness_delta_10f: Optional[float] = None
    thm_mean_delta_10f: Optional[float] = None

    # ── RUNTIME ONLY — None khi offline, KHÔNG train ─────────────────────────
    skew_ms: Optional[float] = None
    fusion_alpha: Optional[float] = None

    # ── METADATA ─────────────────────────────────────────────────────────────
    ts: float = 0.0
    frame_idx: int = 0
    nir_channel: str = "nir"             # "nir" | "rgb"
    thermal_channel: str = "none"         # "lwir" | "none"
    source: str = "rpi"
    thermal_available: bool = False
    motion_available: bool = False
    temporal_available: bool = False

    # ── LABELS ───────────────────────────────────────────────────────────────
    label: Optional[str] = None            # ground truth: dataset gốc hoặc manual annotation
    label_source: Optional[str] = None     # "dataset_original" | "manual" | "weak_heuristic"
    weak_label: Optional[str] = None       # heuristic — chỉ dùng khi dataset không có nhãn
    label_confidence: Optional[float] = None

    def is_compatible_with(self, feature_set: List[str]) -> bool:
        return all(getattr(self, name, None) is not None for name in feature_set)

    def to_feature_array(self, feature_set: List[str]) -> "np.ndarray":
        """Raise nếu feature None. KHÔNG impute."""
        ...

    def effective_label(self) -> Optional[str]:
        """Ưu tiên label (ground truth) trước weak_label.
        Với datasets đã có nhãn gốc: label luôn có giá trị, weak_label không cần thiết.
        """
        return self.label or self.weak_label
```

### 4.3 Availability Rules

| Feature Group | Điều kiện | Nếu không thỏa |
|--------------|-----------|----------------|
| CORE (10) | Luôn thỏa | Record bị reject |
| THERMAL (6) | `thermal_channel = "lwir"` | `None` — record vẫn dùng được cho optical_only |
| MOTION (2) | Video sequence ≥2 frames | `None` — motion_available = False |
| TEMPORAL (2) | ≥10 frame window | `None` — temporal_available = False |
| RUNTIME_ONLY (2) | Source = `"rpi"` | `None` — KHÔNG train |

---

## 5. Dataset Pipeline Design

### 5.1 DatasetSourceBase

```python
class DatasetSourceBase(ABC):
    nir_channel: str          # "nir" | "rgb"
    thermal_channel: str      # "lwir" | "none"
    has_motion: bool
    has_temporal: bool
    source_tag: str
    label_mapping_key: str    # key trong label_mapping.yaml; None nếu không có nhãn gốc

    @abstractmethod
    def iter_frames(
        self,
    ) -> Iterator[Tuple[np.ndarray, Optional[np.ndarray], float, Optional[str], bool]]:
        """
        Yields:
            image_bgr: ndarray BGR
            thermal_raw: ndarray | None
            timestamp: float
            original_label: str | None   ← nhãn gốc của dataset (nếu có)
            sequence_reset: bool
        """
        ...
```

### 5.2 PRIMARY Sources — Weather Datasets

#### Image2WeatherSource

```
Dataset: Image2Weather
nir_channel = "rgb"
thermal_channel = "none"
has_motion = False          ← still images
has_temporal = False
source_tag = "offline_image2weather"
label_mapping_key = "image2weather"

Labels: Sunny / Cloudy / Rainy / Foggy / Snowy
  → Mapped sang ENV class qua label_mapping.yaml
  → label_source = "dataset_original"  ← KHÔNG weak heuristic
  → Snowy records: label = null → bị loại khỏi training trừ khi có nhu cầu

Folder structure (expected):
  data/weather/image2weather/
  ├── Sunny/*.jpg
  ├── Cloudy/*.jpg
  ├── Rainy/*.jpg
  ├── Foggy/*.jpg
  └── Snowy/*.jpg
```

#### WeatherTimeSource

```
Dataset: Weather-Time Classification with Road Images
nir_channel = "rgb"
thermal_channel = "none"
has_motion = False
source_tag = "offline_weather_time"
label_mapping_key = "weather_time"

Labels: joint (weather, time_of_day)
  → Kết hợp mapping weather + time_of_day để suy ra ENV class
  → Ví dụ: (Sunny, Night) → normal_night; (Rainy, Morning) → rain
  → label_source = "dataset_original"

Folder structure (expected):
  data/weather/weather_time/
  ├── Cloudy/
  │   ├── Dawn/*.jpg
  │   ├── Night/*.jpg
  │   └── ...
  ├── Rainy/
  └── ...
```

#### MWDSource (supplement)

```
Dataset: Multi-class Weather Dataset
nir_channel = "rgb"
thermal_channel = "none"
source_tag = "offline_mwd"
label_mapping_key = "mwd"
~1,125 ảnh, 4 lớp
→ Dùng để bổ sung volume cho lớp ít samples; không làm backbone
```

#### Weather11ClassSource (supplement)

```
Dataset: Weather Image Recognition (11 classes)
nir_channel = "rgb"
thermal_channel = "none"
source_tag = "offline_weather11"
label_mapping_key = "weather_11class"
~6,862 ảnh
→ Chỉ giữ lại Fog/Rain/Sandstorm; skip các lớp còn lại (null mapping)
→ Bổ sung fog coverage
```

### 5.3 OPTIONAL Sources — Thermal Context (LLVIP, KAIST)

Chỉ sử dụng khi:
- Cần bổ sung NIR night scene coverage (LLVIP)
- Cần validate thermal feature extraction pipeline (KAIST)
- **Không** dùng làm backbone data cho ENV labels

#### LLVIPNIRSource (optional)

```
nir_channel = "nir"
thermal_channel = "none"
has_motion = False
source_tag = "offline_llvip_nir"
Labels: KHÔNG CÓ ENV labels → weak_label từ heuristic
  → label_source = "weak_heuristic"
  → Chỉ hữu ích để bổ sung "night_clear" examples (LLVIP là ảnh đêm NIR)

Heuristic rule (phải ghi rõ, không để ngầm định):
  - Tất cả ảnh LLVIP đều là night scenes ngoài trời → weak_label = "night_clear"
  - label_confidence = 0.60  ← thấp vì không phân biệt night_clear vs normal_night
  - Cảnh báo: không filter weak_heuristic records này vào optical_only training
    trừ khi explicitly muốn bổ sung night_clear coverage và đã validate distribution
```

#### KAISTSource (optional)

```
nir_channel = "rgb"
thermal_channel = "lwir"
has_motion = True
has_temporal = True
source_tag = "offline_kaist"
Labels: KHÔNG CÓ ENV labels → weak_label từ heuristic
  → label_source = "weak_heuristic"
  → Chỉ dùng để validate thermal feature pipeline và with_thermal ablation
```

### 5.4 Field Collection (Phase 3B-2)

Dùng RPi IMX290 (NIR) để thu thập data cho các classes **không có trong weather datasets**:

| Target ENV class | Cách thu thập | Target samples |
|------------------|--------------|---------------|
| `glare` | Hướng camera vào đèn pha / nguồn sáng điểm | ≥200 |
| `backlight` | Camera ngược sáng (mặt trời / đèn phía sau) | ≥200 |
| `indoor` | Trong nhà, ánh sáng nhân tạo | ≥200 |
| `night_clear` | Đêm ngoài trời, không đèn đường | ≥200 |

Annotation: qua `tools/label_session.py`, `label_source = "manual"`

---

## 6. Tools Specification

### tools/offline_pipeline.py

**Mục đích:** Runner chính — extract features từ datasets, map labels, output JSONL.

**Input:**
```
--dataset    [image2weather | weather_time | mwd | weather11 | llvip_nir | kaist]
--input-dir  <path>
--output     <path.jsonl>
--mapping    tools/label_mapping.yaml     (default)
--interval   <N>                          (default: 1 cho still, 5 cho video)
--skip-null-labels                        (bỏ records với null mapping)
```

**Behavior:**
- Với PRIMARY datasets: `label` = mapped ENV class từ nhãn gốc, `label_source = "dataset_original"`
- Với OPTIONAL datasets (LLVIP/KAIST): `weak_label` từ heuristic, `label_source = "weak_heuristic"`
- Không giả lập temporal cho still images

**CLI:**
```bash
# Primary: Image2Weather
python tools/offline_pipeline.py \
    --dataset image2weather \
    --input-dir data/weather/image2weather \
    --output logs/ml/offline_image2weather.jsonl \
    --skip-null-labels

# Primary: Weather-Time
python tools/offline_pipeline.py \
    --dataset weather_time \
    --input-dir data/weather/weather_time \
    --output logs/ml/offline_weather_time.jsonl \
    --skip-null-labels

# Optional: KAIST (thermal path ablation)
python tools/offline_pipeline.py \
    --dataset kaist \
    --input-dir data/thermal_optional/KAIST \
    --output logs/ml/offline_kaist.jsonl
```

**Dependencies:** `feature_schema.py`, `feature_extractor.py`, `utils.py` (MLLogger), `main.py` (build_frame_cache), `tools/label_mapping.yaml`

---

### tools/validate_schema.py

**Mục đích:** Kiểm tra JSONL file đúng FeatureRecord schema. Chạy sau mỗi extraction.

**Checks:**
1. Mọi record có đủ CORE features (không None, không NaN)
2. `nir_channel` ∈ `{"nir", "rgb"}`
3. `thermal_channel` ∈ `{"lwir", "none"}`
4. Nếu `thermal_available=True` → thermal fields ≠ None
5. Nếu `temporal_available=True` → temporal fields ≠ None
6. Offline records: `skew_ms = None`, `fusion_alpha = None` (bắt buộc)
7. `label_source` ∈ `{"dataset_original", "manual", "weak_heuristic", null}`
8. Nếu `label_source = "dataset_original"` → `label` phải có giá trị
9. `label` nếu có → phải nằm trong `ENV_CLASSES`

```bash
python tools/validate_schema.py --input logs/ml/offline_image2weather.jsonl
# PASS: 180000 records, 0 errors
```

---

### tools/label_session.py

**Mục đích:** Manual annotation cho field collection (glare, indoor, backlight, night_clear).

**Modes:**
```bash
# Interactive per-record (field collection)
python tools/label_session.py \
    --input logs/ml/field_nir_20260410.jsonl \
    --output logs/ml/field_nir_20260410_labeled.jsonl \
    --mode cli

# Bulk assign (toàn bộ session cùng label)
python tools/label_session.py \
    --input logs/ml/field_glare_session1.jsonl \
    --output logs/ml/field_glare_session1_labeled.jsonl \
    --assign glare
```

**Output:** Records với `label` = ENV class, `label_source = "manual"`

---

### tools/check_features.py

**Mục đích:** Phân tích distribution, data quality report.

**Report bao gồm:**
1. Records tổng per source, per nir_channel, per ENV class
2. **Class balance** cho toàn training set (quan trọng — phát hiện imbalance sớm)
3. Zero-variance check cho CORE features → FAIL nếu phát hiện
4. % records có `label` (dataset_original + manual) vs chỉ có `weak_label`
5. Feature range sanity check (nir_mean_brightness ∈ [0, 255], ...)
6. Optional: histogram per feature grouped by nir_channel

```bash
python tools/check_features.py \
    --input logs/ml/offline_image2weather.jsonl \
             logs/ml/offline_weather_time.jsonl \
             logs/ml/field_*.jsonl \
    --plots output/feature_analysis/ \
    --fail-on-zero-variance \
    --target-min-per-class 300
```

---

### tools/mix_datasets.py

**Mục đích:** Merge JSONL, filter, tạo dataset sạch cho training.

**Key filters:**
```
--label-source [dataset_original | manual | weak_heuristic]
--labeled-only              chỉ records có label hoặc weak_label
--min-confidence <float>    chỉ áp dụng khi label_source = "weak_heuristic"
--require-label-source dataset_original,manual   (chỉ dùng nhãn chất lượng cao)
--max-per-class <int>       balance
--nir-channel <val>
--require-thermal
```

**CLI examples:**
```bash
# Dataset cho optical_only training (PRIMARY: weather datasets + field collection)
python tools/mix_datasets.py \
    --input logs/ml/offline_image2weather.jsonl \
            logs/ml/offline_weather_time.jsonl \
            logs/ml/offline_mwd.jsonl \
            logs/ml/field_*.jsonl \
    --require-label-source dataset_original,manual \
    --output data/training/optical_only.jsonl \
    --max-per-class 2000

# Dataset cho with_thermal ablation (KAIST supplement)
python tools/mix_datasets.py \
    --input data/training/optical_only.jsonl \
            logs/ml/offline_kaist.jsonl \
    --require-thermal \
    --output data/training/optical_thermal_ablation.jsonl
```

---

## 7. Training Strategy (FINAL)

### 7.1 TRAINING_MODES

```python
TRAINING_MODES = {
    "optical_only": {
        # ── PRIMARY PRODUCTION MODE ──────────────────────────────────────────
        "feature_set": FEATURE_SET_OPTICAL_ONLY,   # 11 features
        "allowed_nir_channels": ["nir", "rgb"],     # cả hai OK — normalize riêng
        "allowed_thermal_channels": ["none", "lwir"],
        "requires_thermal": False,                  # thermal KHÔNG yêu cầu
        "requires_motion": False,
        "requires_temporal": False,
        "normalize_by": "nir_channel",              # scaler riêng cho nir vs rgb
        "preferred_label_sources": ["dataset_original", "manual"],
        "offline_achievable": True,
        "deploy_ready": True,                       # ← PRIMARY production candidate
        "graceful_degrade_to": None,                # không cần degrade
        "sources": [
            "offline_image2weather",
            "offline_weather_time",
            "offline_mwd",
            "field_collection",
        ],
        "notes": "Primary production model. Works on ALL hardware configs. "
                 "Optical-only — không cần MI48.",
    },
    "with_thermal": {
        # ── OPTIONAL ENHANCEMENT ─────────────────────────────────────────────
        "feature_set": FEATURE_SET_OPTICAL_THERMAL, # 16 features
        "allowed_nir_channels": ["nir"],             # RPi NIR only (không dùng rgb)
        "allowed_thermal_channels": ["lwir"],
        "requires_thermal": True,
        "requires_motion": False,
        "requires_temporal": False,
        "normalize_by": "fixed",
        "preferred_label_sources": ["manual"],       # RPi field data chính
        "offline_achievable": False,                 # KAIST rgb+lwir != nir+lwir
        "deploy_ready": True,                        # khi MI48 available + RPi validated
        "graceful_degrade_to": "optical_only",       # BẮT BUỘC: degrade khi không có MI48
        "sources": ["rpi"],                          # Phase 3C
        "notes": "Enhancement when MI48 present. MUST gracefully degrade to optical_only "
                 "when thermal unavailable. KHÔNG phải production path trên mọi device.",
    },
    # ── Ablation / research modes ─────────────────────────────────────────────
    "rgb_thermal_ablation": {
        "feature_set": FEATURE_SET_OPTICAL_THERMAL,
        "allowed_nir_channels": ["rgb"],
        "allowed_thermal_channels": ["lwir"],
        "requires_thermal": True,
        "deploy_ready": False,                       # rgb != nir — không deploy
        "sources": ["offline_kaist"],
        "notes": "Ablation only. Validates thermal feature pipeline với KAIST data. "
                 "KHÔNG deploy: rgb channel ≠ NIR channel của RPi.",
    },
}
```

### 7.2 Normalization Rules

```
Quy tắc:
  1. optical_only mode: scaler riêng cho nir_channel="nir" và nir_channel="rgb"
       Inference: chọn scaler theo nir_channel của input
  2. with_thermal / rgb_thermal_ablation: 1 scaler (1 combo duy nhất)
  3. KHÔNG normalize chung records có nir_channel khác nhau

Lưu cùng model: {"rf": rf, "scalers": scalers, "feature_set": ..., "training_mode": ...}
```

### 7.3 Production vs Ablation

| | Production Optical (Phase 3B) | Production Thermal (Phase 3C) | Ablation |
|--|-------------------------------|-------------------------------|----------|
| Mode | `optical_only` | `with_thermal` | `rgb_thermal_ablation` |
| Data | Weather datasets + field | RPi field sessions | KAIST |
| Labels | `dataset_original` + `manual` | `manual` | `weak_heuristic` |
| Deploy | Sau Phase 3B + RPi val. | Sau Phase 3C + RPi val. | **KHÔNG** |
| Output dir | `models/baseline/` → `models/production/` | `models/production/` | `models/ablation/` |
| Success metric | Balanced CV ≥ 0.60 (offline) → ≥ 0.75 (RPi test) | ≥ 0.75 (RPi test) | Informational |

---

## 8. Phase Plan (Updated)

### Phase 3A — Infrastructure [LOCAL] ✅ DONE

**Goal:** Foundation modules, không ảnh hưởng runtime.

| Task | File | Mô tả | Status |
|------|------|-------|--------|
| 3A.1 | `feature_schema.py` | FEATURE_SET_* + FeatureRecord + ENV_CLASSES | ✅ |
| 3A.2 | `utils.py` | `MLLogger` (đã có trong config path) | ✅ |
| 3A.3 | `config.py` | `ML_LOG_ENABLED`, `ML_LOG_DIR`, `ML_LOG_INTERVAL`, `ML_MODEL_PATH` | ✅ |
| 3A.4 | `feature_extractor.py` | FeatureExtractor, optical-first | ✅ |
| 3A.5 | `tools/validate_schema.py` | Schema validator | ✅ |
| 3A.6 | `tools/label_mapping.yaml` | Weather → ENV mapping config | ✅ |
| **3A.7** | **`main.py` (MODIFY)** | **Wire FeatureExtractor + MLLogger** | **🔲 TODO — bước tiếp theo** |

> **3A.7 là bước tiếp theo ngay bây giờ.** Phần còn lại của 3A đã DONE.

---

### Phase 3B — Dataset Pipeline + Optical Baseline [LOCAL] ✅ DONE

#### Phase 3B-1: Weather Datasets (optical, có nhãn thật)

| Task | Mô tả | Status |
|------|-------|--------|
| 3B-1.1 | Download datasets vào `data/weather/` | ✅ (mwd, weather11, weather_time, darkface, ExDark, glare, indoor, backlight) |
| 3B-1.2 | `tools/offline_pipeline.py` — implement tất cả sources | ✅ |
| 3B-1.3 | Extraction → 8 JSONL trong `logs/ml/` | ✅ |
| 3B-1.4 | `validate_schema.py` → cần xác nhận (chạy lệnh dưới) | ⚠️ chưa verify |
| 3B-1.5 | `tools/check_features.py` — class balance, zero-variance | ⚠️ chưa chạy trên final mix |
| 3B-1.6 | `tools/mix_datasets.py` → `data/training/optical_only.jsonl` (11316 mẫu) | ✅ |

#### Phase 3B-2: Field Collection (NIR thực địa, cho classes thiếu)

| Task | Mô tả | Status |
|------|-------|--------|
| 3B-2.1 | `glare` NIR thực địa | ⚠️ TẠM ĐỦ qua offline glare dataset (400 mẫu rgb) — cần NIR field ở 3C |
| 3B-2.2 | `backlight` NIR thực địa | ⚠️ TẠM ĐỦ qua offline backlight (368 mẫu rgb) |
| 3B-2.3 | `indoor` NIR thực địa | ⚠️ TẠM ĐỦ qua indoor_cvpr (1792 mẫu rgb) |
| 3B-2.4 | `night_clear` NIR thực địa | ⚠️ TẠM ĐỦ qua darkface (1602 mẫu rgb) |

> Offline substitutes dùng rgb channel, không phải NIR thực địa. Đủ để train baseline, chưa đủ cho production. NIR field collection thực sự sẽ làm ở Phase 3C.

#### Phase 3B-3: Baseline Training ✅ DONE

| Task | Mô tả | Status |
|------|-------|--------|
| 3B-3.1 | `models/train_classifier.py` — TRAINING_MODES | ✅ |
| 3B-3.2 | Mix → `data/training/optical_only.jsonl` (11316 mẫu, 9 classes) | ✅ |
| 3B-3.3 | Train `optical_only` → `models/baseline/rf_optical_only.joblib` (20MB) | ✅ cv_balanced_acc=**0.6729** |
| 3B-3.4 | rgb_thermal_ablation (KAIST) | 🔲 SKIP — chưa cần |

**Exit criteria Phase 3B — đã xác nhận 2026-04-10:**
- `validate_schema.py` PASS trên 8/8 JSONL: ✅
- ≥300 per class: ✅ (thấp nhất: backlight=368)
- Zero-variance CORE: ✅ (`hour_of_day_*`, `prev_env_class` = hằng số expected với ảnh tĩnh)
- CV accuracy ≥ 0.60: ✅ (**0.6158** trên 11316 mẫu full dataset)
- Models trong `models/baseline/`, không có `with_thermal`: ✅

> ⚠️ **Model stale**: `rf_optical_only.joblib` hiện tại train trên 4806 mẫu (mix cũ, trước khi thêm backlight/exdark/indoor_cvpr). Cần retrain trên 11316 mẫu đầy đủ. CV sẽ là ~0.616 (vẫn pass). Lệnh:
> ```bash
> .venv/bin/python models/train_classifier.py \
>   --mode optical_only \
>   --dataset data/training/optical_only.jsonl \
>   --output models/baseline/rf_optical_only.joblib
> ```

---

### Phase 3C — RPi Wire + Field Collection + Production Deploy [HYBRID/RPi] 🔲 TODO

**Thứ tự ưu tiên:**

| Task | Mô tả | Env | Độ ưu tiên |
|------|-------|-----|-----------|
| **3C.0** | **Wire `FeatureExtractor` + `MLLogger` vào `main.py`** (Step 3A.7) | LOCAL | 🔥 Làm ngay |
| 3C.1 | Enable `ML_LOG_ENABLED=True` trên RPi, test FPS không giảm | RPi | Sau 3C.0 |
| 3C.2 | Field sessions NIR: glare, backlight, indoor, night_clear (≥200 mẫu/class) | RPi | Sau 3C.1 |
| 3C.3 | Rsync `logs/ml/field_*.jsonl` về Mac + manual annotation qua `label_session.py` | LOCAL | Sau 3C.2 |
| 3C.4 | Mix NIR field data vào optical_only → retrain → `models/baseline/rf_optical_only_v2.joblib` | LOCAL | Sau 3C.3 |
| 3C.5 | Validate model v2 trên RPi: balanced accuracy ≥ 0.75 | RPi | Sau 3C.4 |
| 3C.6 | Copy sang `models/production/env_classifier.joblib`, set `ML_MODEL_PATH` | LOCAL | Sau 3C.5 |
| 3C.7 | (Optional) Train `with_thermal` nếu MI48 available | LOCAL | Sau 3C.6 |
| 3C.8 | Wire model inference vào `EnvPresetController.tick()` | LOCAL | Sau 3C.6 |

---

## 9. File-Level Implementation Plan

### FILES TO CREATE

| Path | Mục đích | Depends on |
|------|---------|-----------|
| `src/smartbinocular/feature_schema.py` | Feature sets, FeatureRecord, ENV_CLASSES | stdlib, numpy |
| `src/smartbinocular/feature_extractor.py` | FeatureExtractor — optical-first | feature_schema, config |
| `tools/__init__.py` | Package marker | — |
| `tools/label_mapping.yaml` | Weather → ENV mapping config | — |
| `tools/offline_pipeline.py` | Dataset runners (Weather, LLVIP, KAIST) | feature_schema, feature_extractor, utils, label_mapping.yaml |
| `tools/validate_schema.py` | JSONL validator | feature_schema |
| `tools/label_session.py` | Manual annotation | feature_schema |
| `tools/check_features.py` | Distribution analysis | feature_schema, numpy, matplotlib |
| `tools/mix_datasets.py` | Merge + filter | feature_schema |
| `models/__init__.py` | Package marker | — |
| `models/train_classifier.py` | Training pipeline + TRAINING_MODES | feature_schema, sklearn, joblib |
| `data/*`, `models/baseline/` | Nội dung cục bộ / artifact — không dùng `.gitkeep`; xem root `.gitignore`, `data/README.md` | — |

### FILES TO MODIFY

| Path | Thay đổi cụ thể |
|------|----------------|
| `src/smartbinocular/utils.py` | Thêm `MLLogger`: thread-safe async JSONL writer, `log(record: FeatureRecord)`, `close()` |
| `src/smartbinocular/config.py` | Thêm: `ML_LOG_ENABLED: bool = False`, `ML_LOG_DIR: str = "logs/ml"`, `ML_LOG_INTERVAL: int = 5` |
| `src/smartbinocular/main.py` | Wire FeatureExtractor + MLLogger. Wrapped trong `if ML_LOG_ENABLED`. Thermal features chỉ extract khi `thermal_available`. |
| `legacy/md/ML_HYBRID_SYSTEM_PLAN.md` | Thêm note: "Phase 3 restructured — xem `OFFLINE_ML_PLAN.md` v3.0" |
| `pyproject.toml` | Optional deps group `[ml-tools]`: scikit-learn, matplotlib, tqdm, joblib, pyyaml |

### FILES NOT TO TOUCH

```
src/smartbinocular/hardware.py
src/smartbinocular/thermal_pipeline.py   ← read-only import từ feature_extractor
src/smartbinocular/nir_pipeline.py       ← read-only import từ feature_extractor
src/smartbinocular/display_pipeline.py
src/smartbinocular/motion.py             ← read-only import từ feature_extractor
src/smartbinocular/env_presets.py
src/smartbinocular/metrics.py
legacy/md/
```

---

## 10. Execution Order

```
Step 1: src/smartbinocular/feature_schema.py  [NEW]
        Test: python -c "from smartbinocular.feature_schema import FEATURE_SET_OPTICAL_ONLY; print(len(FEATURE_SET_OPTICAL_ONLY))"
        Expected: 10

Step 2: src/smartbinocular/utils.py  [MODIFY — add MLLogger]
        Test: unit test MLLogger với 5 mock FeatureRecord → verify JSONL output

Step 3: src/smartbinocular/config.py  [MODIFY — ML keys]
        Test: python -c "from smartbinocular.config import CONFIG; assert 'ML_LOG_ENABLED' in CONFIG"

Step 4: tools/label_mapping.yaml  [NEW]
        Test: python -c "import yaml; d = yaml.safe_load(open('tools/label_mapping.yaml')); print(list(d.keys()))"
        Expected: ['image2weather', 'weather_time', 'mwd', 'weather_11class']

Step 5: src/smartbinocular/feature_extractor.py  [NEW]
        Test: khởi tạo FeatureExtractor với mock FrameCache
              → FeatureRecord với CORE filled, THERMAL = None (thermal_channel="none")
              → is_compatible_with(FEATURE_SET_OPTICAL_ONLY) = True

Step 6: tools/validate_schema.py  [NEW]
        Test 1: validate mock JSONL 5 records hợp lệ → PASS
        Test 2: validate record có skew_ms ≠ None với source != "rpi" → FAIL

Step 7: tools/offline_pipeline.py  [NEW]
        Test với 10 ảnh Image2Weather:
          → JSONL output có label = ENV class, label_source = "dataset_original"
          → validate_schema.py PASS
          → thermal fields tất cả None (vì weather datasets không có thermal)

Step 8: tools/label_session.py  [NEW]
        Test: bulk-assign "glare" cho JSONL nhỏ → verify label field

Step 9: tools/check_features.py  [NEW]
        Test: chạy trên output Step 7 → report không crash, zero-variance = 0

Step 10: tools/mix_datasets.py  [NEW]
         Test: merge 2 JSONL với --require-label-source dataset_original,manual
               → chỉ records có label (không weak_label-only)

Step 11: models/train_classifier.py  [NEW]
         Test: train optical_only với 50 mock records
               → output trong models/baseline/, metadata ghi "deploy_ready": false

Step 12: src/smartbinocular/main.py  [MODIFY — LAST]
         Implement: additive, wrapped trong if ML_LOG_ENABLED
         Validate RPi: FPS không giảm quá 2 FPS
         Validate: JSONL output có thermal fields = None khi MI48 không có
```

---

## 11. Constraints Summary

| # | Constraint | Lý do |
|---|-----------|-------|
| C1 | Optical pipeline độc lập, không phụ thuộc thermal | Deploy trên mọi hardware config |
| C2 | `optical_only` = PRIMARY production | Thermal là enhancement, không core |
| C3 | `with_thermal` phải graceful degrade về `optical_only` | Không khóa người dùng vào 1 hardware |
| C4 | PRIMARY data: Image2Weather + Weather-Time | Optical-first, có nhãn thật |
| C5 | LLVIP/KAIST = OPTIONAL | Không làm backbone ENV data |
| C6 | Datasets có nhãn gốc: dùng nhãn gốc, không weak heuristic | Tránh làm giảm label quality |
| C7 | Không deploy dataset-only model lên production | Cần RPi validation |
| C8 | Temporal = None với still images | Không giả lập temporal |
| C9 | Không impute zero | Spurious pattern trong RF |
| C10 | Normalize per nir_channel | RGB vs NIR distribution khác nhau |
| C11 | Không có đoạn nào mô tả hệ thống là "multispectral-core" | Hệ thống là optical night vision |
