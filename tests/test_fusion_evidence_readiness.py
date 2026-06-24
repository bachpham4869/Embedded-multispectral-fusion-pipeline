from __future__ import annotations

from pathlib import Path

from tools.harden_fusion_evidence import build_evidence_readiness_rows


def test_evidence_readiness_marks_proxy_fusion_and_runtime_cost(tmp_path: Path):
    artifacts_dir = tmp_path / "artifacts" / "fusion_eval"
    tables_dir = tmp_path / "docs" / "tables" / "fusion"
    figures_dir = tmp_path / "docs" / "figures" / "fusion"
    docs_dir = tmp_path / "docs" / "fusion"
    for path in (artifacts_dir, tables_dir, figures_dir, docs_dir):
        path.mkdir(parents=True)

    (artifacts_dir / "pairing_manifest.csv").write_text(
        "pair_id,pair_status,evidence_label\n"
        "p1,reject_unpaired,unpaired\n"
        "p2,qualitative_weak,proxy\n",
        encoding="utf-8",
    )
    (artifacts_dir / "run_manifest.json").write_text('{"git_commit":"abc"}', encoding="utf-8")
    (tables_dir / "fusion_algorithm_comparison.md").write_text("# Fusion\n", encoding="utf-8")
    (tables_dir / "runtime_timing_summary.md").write_text("# Timing\n", encoding="utf-8")
    (figures_dir / "failure_cases_grid.png").write_bytes(b"small")

    rows = build_evidence_readiness_rows(
        repo_root=tmp_path,
        artifacts_dir=artifacts_dir,
        docs_dir=docs_dir,
        tables_dir=tables_dir,
        figures_dir=figures_dir,
    )

    by_artifact = {row["artifact"]: row for row in rows}
    fusion_row = by_artifact["docs/tables/fusion/fusion_algorithm_comparison.md"]
    assert fusion_row["metric_tier"] == "Tier 3"
    assert fusion_row["evidence_label"] == "proxy/unpaired"
    assert "strict paired = 0" in fusion_row["caveat"]

    runtime_row = by_artifact["docs/tables/fusion/runtime_timing_summary.md"]
    assert runtime_row["metric_tier"] == "Tier 1"
    assert runtime_row["input_data_type"] == "real session JSON"
    assert "runtime cost" in runtime_row["thesis_usability"]
