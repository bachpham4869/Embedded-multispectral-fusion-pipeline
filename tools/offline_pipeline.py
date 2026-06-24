#!/usr/bin/env python3
"""offline_pipeline.py — Extract features from weather/thermal datasets into JSONL.

Usage:
    python tools/offline_pipeline.py \\
        --dataset image2weather \\
        --input-dir data/weather/image2weather \\
        --output logs/ml/offline_image2weather.jsonl \\
        --skip-null-labels

    python tools/offline_pipeline.py \\
        --dataset weather_time \\
        --input-dir data/weather/weather_time \\
        --output logs/ml/offline_weather_time.jsonl \\
        --skip-null-labels

    python tools/offline_pipeline.py \\
        --dataset mwd \\
        --input-dir data/weather/mwd \\
        --output logs/ml/offline_mwd.jsonl

    python tools/offline_pipeline.py \\
        --dataset weather11 \\
        --input-dir data/weather/weather11 \\
        --output logs/ml/offline_weather11.jsonl \\
        --skip-null-labels

Supported datasets: image2weather, weather_time, mwd, weather11, llvip_nir,
    darkface, exdark_street, glare_street, backlight, gray_nir
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Iterator, Optional, TextIO, Tuple

import cv2 as cv
import numpy as np
import yaml

# Allow running from repo root without pip install
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from smartbinocular.feature_extractor import FeatureExtractor
from smartbinocular.feature_schema import ENV_CLASSES, FeatureRecord
from smartbinocular.utils import build_frame_cache


_ENV_CLASS_SET = set(ENV_CLASSES)
_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# Datasets that subsample with shuffle + jittered cap (see --max-env-samples).
_ENV_SUBSAMPLED_DATASETS = frozenset(
    {"darkface", "exdark_street", "glare_street", "backlight", "gray_nir"}
)
_DEFAULT_ENV_CAP = 3000


def _jittered_cap(upper: int, rng: random.Random) -> int:
    """Return a cap ≤ `upper`, with random slack so totals are not round numbers."""
    if upper <= 0:
        return 0
    if upper == 1:
        return 1
    slack = rng.randint(max(1, upper // 28), max(2, upper // 4))
    return max(1, min(upper, upper - slack))


def _safe_label_filename(label: str) -> str:
    """Filesystem-safe stem for per-class JSONL (ENV class names are usually safe)."""
    s = "".join(c if (c.isalnum() or c in "._-") else "_" for c in label.strip())
    return s or "unlabeled"


def _rng_for_cap_jitter(cap_seed: Optional[int]) -> random.Random:
    """Cap jitter: use OS entropy by default so each run gets a different slack (not e.g. always 1602).

    Pass an int for reproducible cap (e.g. ``--cap-seed 42`` to match a fixed experiment).
    """
    if cap_seed is not None:
        return random.Random(cap_seed)
    return random.SystemRandom()


def _resolve_mwd_root(root: Path) -> Path:
    """Kaggle MWD: images live under dataset2/dataset2/. Accept either that path or repo root."""
    root = Path(root).resolve()
    nested = root / "dataset2" / "dataset2"
    if nested.is_dir():
        return nested
    return root


def _resolve_weather11_root(root: Path) -> Path:
    """Kaggle weather-dataset: class folders under dataset/. Accept .../weather11 or .../weather11/dataset."""
    root = Path(root).resolve()
    nested = root / "dataset"
    if nested.is_dir():
        return nested
    return root


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _dhash_bgr(bgr: np.ndarray, hash_size: int = 8) -> str:
    gray = cv.cvtColor(bgr, cv.COLOR_BGR2GRAY)
    resized = cv.resize(gray, (hash_size + 1, hash_size), interpolation=cv.INTER_AREA)
    diff = resized[:, 1:] > resized[:, :-1]
    value = 0
    for bit in diff.flatten():
        value = (value << 1) | int(bool(bit))
    return f"{value:0{hash_size * hash_size // 4}x}"


def _repo_relative(path: Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(Path(__file__).resolve().parents[1]))
    except ValueError:
        return str(path)


# Frames type: (bgr_image, thermal_raw_or_None, timestamp, original_label_or_None, sequence_reset)
FrameYield = Tuple[np.ndarray, Optional[np.ndarray], float, Optional[str], bool]


# ── Label mapping helpers ──────────────────────────────────────────────────────

def load_mapping(mapping_path: Path) -> dict:
    with open(mapping_path) as f:
        return yaml.safe_load(f)


def resolve_label(mapping_key: str, raw_label: str, mapping: dict) -> Tuple[Optional[str], float, str]:
    """Return (env_class_or_None, confidence, label_type).

    label_type is one of "dataset_original", "weak_heuristic".
    Returns (None, 0.0, "dataset_original") if mapping says null (skip record).
    """
    section = mapping.get(mapping_key, {})
    entry = section.get(raw_label, {})
    if not entry:
        return None, 0.0, "dataset_original"
    env = entry.get("env")
    conf = float(entry.get("confidence", 0.5))
    if env in _ENV_CLASS_SET:
        return env, conf, "dataset_original"
    return None, conf, "dataset_original"  # null → skip


def resolve_weather_time_label(
    weather: str, period: str, mapping: dict
) -> Tuple[Optional[str], float]:
    """Resolve combined weather+period label for weather_time dataset."""
    wt_section = mapping.get("weather_time", {})

    # Check combined_overrides first (period takes priority over weather)
    for override in wt_section.get("combined_overrides", []):
        match = override.get("match", {})
        if all(
            (k == "period" and period == v) or (k == "weather" and weather == v)
            for k, v in match.items()
        ):
            env = override.get("env")
            conf = float(override.get("confidence", 0.75))
            if env in _ENV_CLASS_SET:
                return env, conf
            return None, conf

    # Fall back to weather label
    weather_section = wt_section.get("weather", {})
    entry = weather_section.get(weather, {})
    env = entry.get("env")
    conf = float(entry.get("confidence", 0.5))
    if env in _ENV_CLASS_SET:
        return env, conf
    return None, conf


# ── Dataset source base ────────────────────────────────────────────────────────

class DatasetSourceBase(ABC):
    nir_channel: str = "rgb"
    thermal_channel: str = "none"
    has_motion: bool = False
    has_temporal: bool = False
    source_tag: str = "offline_unknown"
    label_mapping_key: Optional[str] = None
    _last_image_path: Optional[Path] = None

    @abstractmethod
    def iter_frames(self) -> Iterator[FrameYield]:
        """Yield (bgr, thermal_or_None, ts, original_label_or_None, sequence_reset)."""
        ...

    def _load_image(self, path: Path) -> Optional[np.ndarray]:
        # Decode by file contents — Kaggle MWD has a few *.jpg files that are actually GIFs;
        # cv.imread() uses the extension and returns None for those.
        p = Path(path)
        data = np.fromfile(str(p), dtype=np.uint8)
        img = cv.imdecode(data, cv.IMREAD_COLOR)
        if img is None:
            img = cv.imread(str(p), cv.IMREAD_COLOR)
        if img is None:
            print(f"  WARN: Could not read image: {path}", file=sys.stderr)
        else:
            self._last_image_path = p
        return img

    def _dummy_thermal(self) -> np.ndarray:
        """Return all-zero thermal for optical-only datasets."""
        return np.zeros((62, 80), dtype=np.uint8)


# ── Image2Weather ─────────────────────────────────────────────────────────────

class Image2WeatherSource(DatasetSourceBase):
    """Layout: {root}/{ClassName}/*.jpg  (Sunny, Cloudy, Rainy, Foggy, Snowy)"""
    nir_channel = "rgb"
    thermal_channel = "none"
    has_motion = False
    has_temporal = False
    source_tag = "offline_image2weather"
    label_mapping_key = "image2weather"

    def __init__(self, root: Path):
        self.root = Path(root)

    def iter_frames(self) -> Iterator[FrameYield]:
        for class_dir in sorted(self.root.iterdir()):
            if not class_dir.is_dir():
                continue
            class_name = class_dir.name
            for img_path in sorted(class_dir.iterdir()):
                if img_path.suffix.lower() not in _IMG_EXTS:
                    continue
                bgr = self._load_image(img_path)
                if bgr is None:
                    continue
                # sequence_reset=True for every image (stills — no temporal context)
                yield bgr, None, 0.0, class_name, True


# ── Weather-Time ──────────────────────────────────────────────────────────────

class WeatherTimeSource(DatasetSourceBase):
    """Layout: JSON file with annotations; images in train_dataset/train_images/"""
    nir_channel = "rgb"
    thermal_channel = "none"
    has_motion = False
    has_temporal = False
    source_tag = "offline_weather_time"
    label_mapping_key = "weather_time"

    def __init__(self, root: Path):
        self.root = Path(root)

    def _load_annotations(self) -> list:
        # Try train_dataset/train.json first, then root-level train.json
        candidates = [
            self.root / "train_dataset" / "train.json",
            self.root / "train.json",
        ]
        for p in candidates:
            if p.exists():
                with open(p) as f:
                    data = json.load(f)
                if isinstance(data, dict) and "annotations" in data:
                    return data["annotations"]
                if isinstance(data, list):
                    return data
        raise FileNotFoundError(f"Cannot find train.json under {self.root}")

    def iter_frames(self) -> Iterator[FrameYield]:
        annotations = self._load_annotations()
        # Find the images base dir
        img_base_candidates = [
            self.root / "train_dataset" / "train_images",
            self.root / "train_images",
        ]
        img_base = next((p for p in img_base_candidates if p.exists()), None)
        if img_base is None:
            raise FileNotFoundError(f"Cannot find train_images under {self.root}")

        for ann in annotations:
            fname = ann.get("filename", "")
            # Normalize Windows backslashes and strip leading path component
            fname = fname.replace("\\", "/")
            fname = Path(fname).name  # just the filename
            img_path = img_base / fname
            if not img_path.exists():
                continue
            bgr = self._load_image(img_path)
            if bgr is None:
                continue
            # Encode (weather, period) as a compound label string for mapping
            weather = ann.get("weather", "")
            period = ann.get("period", "")
            compound = f"{weather}|{period}"
            yield bgr, None, 0.0, compound, True


# ── MWD ───────────────────────────────────────────────────────────────────────

_MWD_LABEL_RE = re.compile(r"^([a-z]+)", re.IGNORECASE)


class MWDSource(DatasetSourceBase):
    """Multi-class Weather Dataset. Layout: flat folder, filename prefix = label.
    Known prefixes: cloudy, rain, shine, sunrise
    Kaggle layout: pass data/weather/mwd or .../dataset2/dataset2 (see _resolve_mwd_root).
    """
    nir_channel = "rgb"
    thermal_channel = "none"
    has_motion = False
    has_temporal = False
    source_tag = "offline_mwd"
    label_mapping_key = "mwd"

    def __init__(self, root: Path):
        self.root = _resolve_mwd_root(root)

    def iter_frames(self) -> Iterator[FrameYield]:
        for img_path in sorted(self.root.iterdir()):
            if img_path.suffix.lower() not in _IMG_EXTS:
                continue
            m = _MWD_LABEL_RE.match(img_path.stem)
            if not m:
                continue
            label = m.group(1).lower()
            bgr = self._load_image(img_path)
            if bgr is None:
                continue
            yield bgr, None, 0.0, label, True


# ── Weather11 ─────────────────────────────────────────────────────────────────

class Weather11Source(DatasetSourceBase):
    """11-class Weather. Layout: {root}/{class_name}/*.jpg (class names lowercased).
    Kaggle layout: pass data/weather/weather11 or .../weather11/dataset.
    """
    nir_channel = "rgb"
    thermal_channel = "none"
    has_motion = False
    has_temporal = False
    source_tag = "offline_weather11"
    label_mapping_key = "weather11"

    def __init__(self, root: Path):
        self.root = _resolve_weather11_root(root)

    def iter_frames(self) -> Iterator[FrameYield]:
        for class_dir in sorted(self.root.iterdir()):
            if not class_dir.is_dir():
                continue
            class_name = class_dir.name.lower()
            for img_path in sorted(class_dir.iterdir()):
                if img_path.suffix.lower() not in _IMG_EXTS:
                    continue
                bgr = self._load_image(img_path)
                if bgr is None:
                    continue
                yield bgr, None, 0.0, class_name, True


# ── LLVIP NIR (Optional) ──────────────────────────────────────────────────────

class LLVIPNIRSource(DatasetSourceBase):
    """LLVIP IR images. All images → weak_label = 'night_clear'. No ENV label."""
    nir_channel = "nir"
    thermal_channel = "none"
    has_motion = False
    has_temporal = False
    source_tag = "offline_llvip_nir"
    label_mapping_key = None  # no original labels

    def __init__(self, root: Path):
        self.root = Path(root)

    def iter_frames(self) -> Iterator[FrameYield]:
        for img_path in sorted(self.root.rglob("*")):
            if img_path.suffix.lower() not in _IMG_EXTS:
                continue
            bgr = self._load_image(img_path)
            if bgr is None:
                continue
            yield bgr, None, 0.0, None, True  # original_label=None → weak_heuristic


# ── Dark Face → night_clear ───────────────────────────────────────────────────

class DarkFaceSource(DatasetSourceBase):
    """DarkFace: very dark faces / night. Layout: {root}/image/*.{png,jpg,...}

    Synthetic label ``all`` → map to ``night_clear`` in label_mapping.yaml.
    """
    nir_channel = "rgb"
    thermal_channel = "none"
    has_motion = False
    has_temporal = False
    source_tag = "offline_darkface"
    label_mapping_key = "darkface"

    def __init__(self, root: Path, max_files: Optional[int] = None, rng_seed: int = 42):
        self.root = Path(root)
        nested = self.root / "image"
        self.image_root = nested if nested.is_dir() else self.root
        self.max_files = max_files
        self.rng_seed = rng_seed

    def iter_frames(self) -> Iterator[FrameYield]:
        rng = random.Random(self.rng_seed)
        paths = [p for p in self.image_root.rglob("*") if p.suffix.lower() in _IMG_EXTS]
        rng.shuffle(paths)
        if self.max_files is not None:
            paths = paths[: self.max_files]
        for img_path in paths:
            bgr = self._load_image(img_path)
            if bgr is None:
                continue
            yield bgr, None, 0.0, "all", True


# ── Backlight (flat folder → backlight) ──────────────────────────────────────

class BacklightSource(DatasetSourceBase):
    """Backlit-scene images in a single directory (or nested). Synthetic label ``all``.

    Typical layout: ``data/weather/backlight/*.JPG`` — map ``all`` → ``backlight`` in label_mapping.yaml.
    """
    nir_channel = "rgb"
    thermal_channel = "none"
    has_motion = False
    has_temporal = False
    source_tag = "offline_backlight"
    label_mapping_key = "backlight"

    def __init__(self, root: Path, max_files: Optional[int] = None, rng_seed: int = 42):
        self.root = Path(root)
        self.max_files = max_files
        self.rng_seed = rng_seed

    def iter_frames(self) -> Iterator[FrameYield]:
        rng = random.Random(self.rng_seed)
        paths = [p for p in self.root.rglob("*") if p.is_file() and p.suffix.lower() in _IMG_EXTS]
        rng.shuffle(paths)
        if self.max_files is not None:
            paths = paths[: self.max_files]
        for img_path in paths:
            bgr = self._load_image(img_path)
            if bgr is None:
                continue
            yield bgr, None, 0.0, "all", True


# ── Gray / IMX290 NIR-assisted (field grayscale-like mono) ───────────────────

class GrayNIRSource(DatasetSourceBase):
    """Field NIR-dominant mono (e.g. IMX290 in dark + NIR assist). ENV label ``nir_night``.

    Layout: ``data/weather/gray/**/*.{jpg,png,...}`` — synthetic key ``all`` → ``gray_nir`` in
    ``label_mapping.yaml``. Uses ``nir_channel=rgb`` for scaler grouping (C10).
    """
    nir_channel = "rgb"
    thermal_channel = "none"
    has_motion = False
    has_temporal = False
    source_tag = "offline_gray_nir"
    label_mapping_key = "gray_nir"

    def __init__(self, root: Path, max_files: Optional[int] = None, rng_seed: int = 42):
        self.root = Path(root)
        self.max_files = max_files
        self.rng_seed = rng_seed

    def iter_frames(self) -> Iterator[FrameYield]:
        rng = random.Random(self.rng_seed)
        paths = [p for p in self.root.rglob("*") if p.is_file() and p.suffix.lower() in _IMG_EXTS]
        rng.shuffle(paths)
        if self.max_files is not None:
            paths = paths[: self.max_files]
        for img_path in paths:
            bgr = self._load_image(img_path)
            if bgr is None:
                continue
            yield bgr, None, 0.0, "all", True


# ── ExDark street (Boat, Bus, Car, Motorbike) → normal_night ──────────────────

_EXDARK_STREET = frozenset({"boat", "bus", "car", "motorbike"})


class ExDarkStreetSource(DatasetSourceBase):
    """ExDark: only street vehicle folders → ``normal_night``.

    Pass ``--input-dir`` = ExDark root (folder contains Boat/, Bus/, …).
    """
    nir_channel = "rgb"
    thermal_channel = "none"
    has_motion = False
    has_temporal = False
    source_tag = "offline_exdark_street"
    label_mapping_key = "exdark_street"

    def __init__(self, root: Path, max_files: Optional[int] = None, rng_seed: int = 42):
        self.root = Path(root)
        self.max_files = max_files
        self.rng_seed = rng_seed

    def iter_frames(self) -> Iterator[FrameYield]:
        rng = random.Random(self.rng_seed)
        pairs: list[Tuple[Path, str]] = []
        for class_dir in self.root.iterdir():
            if not class_dir.is_dir():
                continue
            key = class_dir.name.lower()
            if key not in _EXDARK_STREET:
                continue
            for img_path in class_dir.iterdir():
                if img_path.suffix.lower() not in _IMG_EXTS:
                    continue
                pairs.append((img_path, key))
        rng.shuffle(pairs)
        if self.max_files is not None:
            pairs = pairs[: self.max_files]
        for img_path, key in pairs:
            bgr = self._load_image(img_path)
            if bgr is None:
                continue
            yield bgr, None, 0.0, key, True


def _path_has_mask_segment(path: Path) -> bool:
    return any(part.lower() == "mask" for part in path.parts)


# ── Glare (street lights; exclude mask folders) → glare ───────────────────────

class GlareStreetSource(DatasetSourceBase):
    """Glare dataset: all images under root except any path segment named ``mask``."""
    nir_channel = "rgb"
    thermal_channel = "none"
    has_motion = False
    has_temporal = False
    source_tag = "offline_glare_street"
    label_mapping_key = "glare_street"

    def __init__(self, root: Path, max_files: Optional[int] = None, rng_seed: int = 42):
        self.root = Path(root)
        self.max_files = max_files
        self.rng_seed = rng_seed

    def iter_frames(self) -> Iterator[FrameYield]:
        rng = random.Random(self.rng_seed)
        paths: list[Path] = []
        for p in self.root.rglob("*"):
            if not p.is_file() or p.suffix.lower() not in _IMG_EXTS:
                continue
            if _path_has_mask_segment(p):
                continue
            paths.append(p)
        rng.shuffle(paths)
        if self.max_files is not None:
            paths = paths[: self.max_files]
        for img_path in paths:
            bgr = self._load_image(img_path)
            if bgr is None:
                continue
            yield bgr, None, 0.0, "glare", True


# ── Pipeline runner ────────────────────────────────────────────────────────────

DATASET_SOURCES = {
    "image2weather": Image2WeatherSource,
    "weather_time": WeatherTimeSource,
    "mwd": MWDSource,
    "weather11": Weather11Source,
    "llvip_nir": LLVIPNIRSource,
    "darkface": DarkFaceSource,
    "exdark_street": ExDarkStreetSource,
    "glare_street": GlareStreetSource,
    "backlight": BacklightSource,
    "gray_nir": GrayNIRSource,
}


def run_pipeline(
    dataset: str,
    input_dir: Path,
    output_path: Path,
    mapping: dict,
    skip_null_labels: bool = True,
    interval: int = 1,
    max_records: Optional[int] = None,
    quiet: bool = False,
    source_kwargs: Optional[dict] = None,
    by_label_dir: Optional[Path] = None,
    emit_source_metadata: bool = False,
) -> int:
    """Run extraction for one dataset. Returns number of records written."""
    SourceClass = DATASET_SOURCES[dataset]
    sk = source_kwargs or {}
    source = SourceClass(input_dir, **sk)
    extractor = FeatureExtractor()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    skipped_null = 0
    skipped_img = 0

    label_handles: Dict[str, TextIO] = {}
    try:
        with open(output_path, "w") as out_f:
            for frame_idx, (bgr, thermal_raw, ts, original_label, seq_reset) in enumerate(
                source.iter_frames()
            ):
                if frame_idx % interval != 0:
                    continue
                if max_records is not None and written >= max_records:
                    break

                # Reset temporal state at sequence boundaries (C7)
                if seq_reset:
                    extractor.reset_sequence()

                # ── Resolve label ─────────────────────────────────────────────────
                label: Optional[str] = None
                weak_label: Optional[str] = None
                label_source: Optional[str] = None
                label_confidence: Optional[float] = None

                if source.label_mapping_key is None:
                    # LLVIP: heuristic only
                    llvip_section = mapping.get("llvip_nir", {}).get("__all__", {})
                    wl = llvip_section.get("weak_label")
                    if wl in _ENV_CLASS_SET:
                        weak_label = wl
                        label_source = "weak_heuristic"
                        label_confidence = float(llvip_section.get("confidence", 0.60))
                elif dataset == "weather_time":
                    # Compound label: "Cloudy|Morning"
                    if original_label and "|" in original_label:
                        weather_str, period_str = original_label.split("|", 1)
                        env, conf = resolve_weather_time_label(weather_str, period_str, mapping)
                        if env:
                            label = env
                            label_source = "dataset_original"
                            label_confidence = conf
                else:
                    if original_label is not None:
                        env, conf, ls = resolve_label(source.label_mapping_key, original_label, mapping)
                        if env:
                            label = env
                            label_source = ls
                            label_confidence = conf

                # Skip records without usable label if requested
                effective = label or weak_label
                if skip_null_labels and effective is None:
                    skipped_null += 1
                    continue

                # ── Build FrameCache ──────────────────────────────────────────────
                # For optical-only datasets, pass dummy thermal (all zeros = invalid)
                thermal_arr = thermal_raw if thermal_raw is not None else np.zeros((62, 80), dtype=np.uint8)
                cache = build_frame_cache(bgr, thermal_arr, ts)

                # ── Extract features ──────────────────────────────────────────────
                record = extractor.extract(
                    cache,
                    nir_channel=source.nir_channel,
                    thermal_channel=source.thermal_channel,
                    ts=ts,
                    source=source.source_tag,
                    frame_idx=written,
                    has_motion=source.has_motion,
                    has_temporal=source.has_temporal,
                    label=label,
                    label_source=label_source,
                    weak_label=weak_label,
                    label_confidence=label_confidence,
                )

                payload = record.to_dict()
                if emit_source_metadata:
                    img_path = getattr(source, "_last_image_path", None)
                    if img_path is not None:
                        rel = _repo_relative(img_path)
                        file_hash = _sha256_file(img_path)
                        payload.update(
                            {
                                "source_dataset": dataset,
                                "original_image_path": rel,
                                "relative_image_id": rel,
                                "file_sha256": file_hash,
                                "dhash": _dhash_bgr(bgr),
                                "split_group_id": f"file_sha256::{file_hash}",
                                "capture_device": "offline_dataset",
                                "metadata_status": "verified",
                                "metadata_recovery_method": "offline_pipeline_emit_source_metadata",
                                "metadata_missing": [],
                            }
                        )
                    else:
                        payload.update(
                            {
                                "source_dataset": dataset,
                                "metadata_status": "unresolved",
                                "metadata_recovery_method": "offline_pipeline_emit_source_metadata",
                                "metadata_missing": ["original image path"],
                            }
                        )

                line = json.dumps(payload) + "\n"
                out_f.write(line)
                written += 1

                if by_label_dir is not None:
                    eff = record.effective_label()
                    stem = _safe_label_filename(eff if eff is not None else "unlabeled")
                    if stem not in label_handles:
                        by_label_dir.mkdir(parents=True, exist_ok=True)
                        label_handles[stem] = open(
                            by_label_dir / f"{stem}.jsonl", "a", encoding="utf-8"
                        )
                    label_handles[stem].write(line)

                if not quiet and written % 500 == 0:
                    print(f"  {written} records written...", flush=True)
    finally:
        for fh in label_handles.values():
            fh.close()

    if not quiet:
        print(f"Done. Written={written}, Skipped(null)={skipped_null}")
    return written


# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract features from weather/NIR datasets into JSONL."
    )
    parser.add_argument(
        "--dataset", required=True,
        choices=list(DATASET_SOURCES.keys()),
        help="Dataset type"
    )
    parser.add_argument(
        "--input-dir", required=True, type=Path,
        help="Root directory of the dataset"
    )
    parser.add_argument(
        "--output", required=True, type=Path,
        help="Output JSONL file path"
    )
    parser.add_argument(
        "--mapping", default=None, type=Path,
        help="Path to label_mapping.yaml (default: tools/label_mapping.yaml)"
    )
    parser.add_argument(
        "--interval", type=int, default=1,
        help="Sample every Nth frame (default: 1 = all frames)"
    )
    parser.add_argument(
        "--skip-null-labels", action="store_true", default=True,
        help="Skip records with no resolvable label (default: True)"
    )
    parser.add_argument(
        "--keep-null-labels", action="store_true",
        help="Keep records with no resolvable label (overrides --skip-null-labels)"
    )
    parser.add_argument(
        "--max-records", type=int, default=None,
        help="Stop after writing this many records"
    )
    parser.add_argument(
        "--max-env-samples", type=int, default=None,
        metavar="N",
        help=(
            "For darkface | exdark_street | glare_street | backlight | gray_nir: shuffle images "
            "then take at most N (default %d), then apply a random slack so counts are not round."
            % _DEFAULT_ENV_CAP
        ),
    )
    parser.add_argument(
        "--sample-seed", type=int, default=42,
        help="RNG seed for shuffling image order in env-subsampled datasets (default: 42)",
    )
    parser.add_argument(
        "--cap-seed",
        type=int,
        default=None,
        metavar="INT",
        help=(
            "Optional seed for jittered env subsample cap only. "
            "Default: omit = cap varies each run (OS entropy). "
            "Set an int for a fixed cap across runs (e.g. 42 reproduces the old single-seed behavior)."
        ),
    )
    parser.add_argument(
        "--by-label-dir",
        type=Path,
        default=None,
        metavar="DIR",
        help=(
            "Also append each record to DIR/<effective_label>.jsonl (append mode; "
            "use a fresh DIR or delete old *.jsonl before re-running all datasets)."
        ),
    )
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--emit-source-metadata",
        action="store_true",
        help=(
            "Append recoverable source-image metadata to JSONL. Default output is unchanged "
            "when this flag is absent."
        ),
    )
    args = parser.parse_args()

    skip = args.skip_null_labels and not args.keep_null_labels

    mapping_path = args.mapping
    if mapping_path is None:
        # Default: tools/label_mapping.yaml relative to repo root
        mapping_path = Path(__file__).parent / "label_mapping.yaml"
    if not mapping_path.exists():
        print(f"ERROR: label_mapping.yaml not found at {mapping_path}", file=sys.stderr)
        sys.exit(1)
    mapping = load_mapping(mapping_path)

    if not args.input_dir.exists():
        print(f"ERROR: input-dir not found: {args.input_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Dataset : {args.dataset}")
    print(f"Input   : {args.input_dir}")
    print(f"Output  : {args.output}")
    if args.by_label_dir is not None:
        print(f"By-label: {args.by_label_dir}  (append per class)")
    print(f"Interval: every {args.interval} frame(s)")
    print(f"Skip null labels: {skip}")
    source_kwargs = None
    if args.dataset in _ENV_SUBSAMPLED_DATASETS:
        upper = args.max_env_samples if args.max_env_samples is not None else _DEFAULT_ENV_CAP
        cap_rng = _rng_for_cap_jitter(args.cap_seed)
        eff_cap = _jittered_cap(upper, cap_rng)
        source_kwargs = {"max_files": eff_cap, "rng_seed": args.sample_seed}
        cap_mode = f"cap_seed={args.cap_seed}" if args.cap_seed is not None else "cap_seed=random (OS)"
        print(
            f"Env subsample cap (jittered): {eff_cap}  "
            f"(upper={upper}, {cap_mode}, shuffle_seed={args.sample_seed})"
        )
    t0 = time.time()

    n = run_pipeline(
        dataset=args.dataset,
        input_dir=args.input_dir,
        output_path=args.output,
        mapping=mapping,
        skip_null_labels=skip,
        interval=args.interval,
        max_records=args.max_records,
        quiet=args.quiet,
        source_kwargs=source_kwargs,
        by_label_dir=args.by_label_dir,
        emit_source_metadata=args.emit_source_metadata,
    )

    elapsed = time.time() - t0
    print(f"Total records: {n}  |  Elapsed: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
