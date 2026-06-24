from __future__ import annotations

from tools.harden_fusion_evidence import (
    REQUIRED_FUSION_MODES,
    REQUIRED_NIR_BUCKETS,
    REQUIRED_THERMAL_STAGES,
    build_processing_name_alias_rows,
)


def test_processing_name_aliases_cover_required_taxonomy_names():
    rows = build_processing_name_alias_rows()
    names = {row["report_name"] for row in rows}

    assert set(REQUIRED_NIR_BUCKETS).issubset(names)
    assert set(REQUIRED_FUSION_MODES).issubset(names)
    assert set(REQUIRED_THERMAL_STAGES).issubset(names)

    for row in rows:
        assert row["report_name"]
        assert row["category"] in {"nir_processing_bucket", "fusion_mode", "thermal_stage"}
        assert row["code_or_artifact_alias"]
        assert row["rename_policy"] == "report alias only; no production symbol rename"
