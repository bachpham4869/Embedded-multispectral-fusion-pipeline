from __future__ import annotations

import json
from pathlib import Path

from tools.fusion_eval_manifest import (
    classify_artifact,
    discover_artifacts,
    format_pairing_manifest_rows,
    pair_capture_records,
    pair_status_for_gap,
)


def _touch_image(path: Path) -> None:
    path.write_bytes(b"not a real image")


def test_pair_status_boundaries():
    assert pair_status_for_gap(None) == "missing_pair"
    assert pair_status_for_gap(0.5) == "strict_paired"
    assert pair_status_for_gap(1.0) == "strict_paired"
    assert pair_status_for_gap(2.0) == "qualitative_weak"
    assert pair_status_for_gap(20.0) == "qualitative_weak"
    assert pair_status_for_gap(21.0) == "reject_unpaired"


def test_classify_artifact_uses_taxonomy_modes():
    assert classify_artifact(Path("auto_fusion_20260508-004422.png")).kind == "fusion_output"
    assert classify_artifact(Path("auto_imx_20260508-004404.png")).kind == "nir_raw"
    assert classify_artifact(Path("auto_thermal_20260508-004442.png")).kind == "thermal_output"
    assert classify_artifact(Path("session_20260508-012858.json")).kind == "session_metrics"


def test_discover_artifacts_reads_sidecar_session_and_timestamp(tmp_path: Path):
    image = tmp_path / "auto_fusion_20260508-004422.png"
    _touch_image(image)
    image.with_suffix(".json").write_text(
        json.dumps({"session_id": "s1", "timestamp_iso": "2026-05-08T00:44:22", "mode": "fusion"}),
        encoding="utf-8",
    )
    records = discover_artifacts([tmp_path])
    image_records = [r for r in records if r.path.name.endswith(".png")]
    assert len(image_records) == 1
    assert image_records[0].session_id == "s1"
    assert image_records[0].timestamp_iso.startswith("2026-05-08")
    assert image_records[0].timestamp_source == "sidecar"


def test_pair_capture_records_marks_strict_and_proxy_limitations(tmp_path: Path):
    fus = tmp_path / "auto_fusion_20260508-004422.png"
    nir = tmp_path / "auto_imx_20260508-004421.png"
    thermal = tmp_path / "auto_thermal_20260508-004421.png"
    for path in (fus, nir, thermal):
        _touch_image(path)
        path.with_suffix(".json").write_text(
            json.dumps({"session_id": "s1", "timestamp_iso": "2026-05-08T00:44:21.500"}),
            encoding="utf-8",
        )

    pairs = pair_capture_records(discover_artifacts([tmp_path]))
    assert len(pairs) == 1
    assert pairs[0].pair_status == "strict_paired"
    assert pairs[0].evidence_label == "real_paired"

    rows = format_pairing_manifest_rows(pairs)
    assert rows[0]["fusion_path"].endswith("auto_fusion_20260508-004422.png")
    assert rows[0]["nir_path"].endswith("auto_imx_20260508-004421.png")
    assert rows[0]["thermal_path"].endswith("auto_thermal_20260508-004421.png")


def test_pair_capture_records_handles_missing_nir_pair(tmp_path: Path):
    fus = tmp_path / "auto_fusion_20260508-004422.png"
    _touch_image(fus)
    fus.with_suffix(".json").write_text(
        json.dumps({"session_id": "s1", "timestamp_iso": "2026-05-08T00:44:22"}),
        encoding="utf-8",
    )

    pairs = pair_capture_records(discover_artifacts([tmp_path]))
    assert len(pairs) == 1
    assert pairs[0].pair_status == "missing_pair"
    assert pairs[0].evidence_label == "unpaired"
    assert pairs[0].nir is None
