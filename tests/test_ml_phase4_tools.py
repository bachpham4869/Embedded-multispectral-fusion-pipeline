from __future__ import annotations

from tools.analyze_dhash_pairs import classify_pair_category
from tools.build_duplicate_clusters import build_duplicate_clusters
from tools.check_image_level_leakage import check_leakage
from tools.split_duplicate_cluster_aware_jsonl import duplicate_cluster_aware_split


def _row(label: str, group: str, cluster_seed: str, **extra: object) -> dict[str, object]:
    row: dict[str, object] = {
        "label": label,
        "source": "fixture",
        "metadata_status": "verified",
        "split_group_id": group,
        "relative_image_id": f"fixture/{cluster_seed}.jpg",
        "original_image_path": f"fixture/{cluster_seed}.jpg",
        "dhash": "0000000000000000",
        "file_sha256": f"sha-{cluster_seed}",
    }
    row.update(extra)
    return row


def test_pair_classification_keeps_exact_near_false_positive_and_unresolved_separate() -> None:
    train = _row("fog", "g1", "a", file_sha256="same")
    test = _row("fog", "g2", "b", file_sha256="same")
    assert classify_pair_category(train, test, hamming_distance=0) == "exact_file_duplicate"

    train = _row("fog", "g1", "a", file_sha256="a")
    test = _row("fog", "g2", "b", file_sha256="b")
    assert classify_pair_category(train, test, hamming_distance=2) == "likely_near_duplicate"

    train = _row("fog", "g1", "a", file_sha256="a")
    test = _row("normal_day", "g2", "b", file_sha256="b")
    assert classify_pair_category(train, test, hamming_distance=4) == "dhash_false_positive_candidate"

    train = _row("fog", "g1", "a", file_sha256="", metadata_status="unresolved")
    test = _row("fog", "g2", "b", file_sha256="")
    assert classify_pair_category(train, test, hamming_distance=1) == "unresolved"


def test_duplicate_clusters_recompute_full_dataset_graph_for_threshold_modes() -> None:
    rows = [
        _row("fog", "g0", "a", dhash="0000000000000000"),
        _row("fog", "g1", "b", dhash="0000000000000003"),
        _row("fog", "g2", "c", dhash="000000000000001f"),
        _row("rain", "g3", "d", dhash="ffffffffffffffff"),
    ]

    strict_rows, strict_summary, strict_edges = build_duplicate_clusters(rows, mode="strict", dhash_threshold=2)
    conservative_rows, conservative_summary, conservative_edges = build_duplicate_clusters(
        rows,
        mode="conservative",
        dhash_threshold=4,
    )

    assert strict_summary["dhash_threshold"] == 2
    assert conservative_summary["dhash_threshold"] == 4
    assert len(strict_edges) == 1
    assert len(conservative_edges) == 2
    assert strict_rows[0]["duplicate_cluster_id"] == strict_rows[1]["duplicate_cluster_id"]
    assert strict_rows[1]["duplicate_cluster_id"] != strict_rows[2]["duplicate_cluster_id"]
    assert conservative_rows[0]["duplicate_cluster_id"] == conservative_rows[2]["duplicate_cluster_id"]


def test_duplicate_clusters_include_exact_file_sha_edges_even_without_dhash() -> None:
    rows = [
        _row("fog", "g0", "a", dhash="", file_sha256="same"),
        _row("fog", "g1", "b", dhash="", file_sha256="same"),
        _row("rain", "g2", "c", dhash="", file_sha256="other"),
    ]

    clustered, summary, edges = build_duplicate_clusters(rows, mode="strict", dhash_threshold=2)

    assert summary["exact_file_sha_edges"] == 1
    assert len(edges) == 1
    assert edges[0]["edge_type"] == "exact_file_duplicate"
    assert clustered[0]["duplicate_cluster_id"] == clustered[1]["duplicate_cluster_id"]
    assert clustered[1]["duplicate_cluster_id"] != clustered[2]["duplicate_cluster_id"]


def test_duplicate_cluster_ids_do_not_collide_on_shared_path_prefixes() -> None:
    rows = [
        _row("glare", "g0", "same_prefix_0001", dhash="0000000000000000"),
        _row("glare", "g1", "same_prefix_0002", dhash="ffffffffffffffff"),
    ]

    clustered, summary, edges = build_duplicate_clusters(rows, mode="strict", dhash_threshold=2)

    assert edges == []
    assert summary["cluster_count"] == 2
    assert clustered[0]["duplicate_cluster_id"] != clustered[1]["duplicate_cluster_id"]


def test_cluster_aware_split_has_zero_duplicate_cluster_and_split_group_overlap() -> None:
    rows = [
        _row("fog", "g0", "a", duplicate_cluster_id="c0"),
        _row("fog", "g1", "b", duplicate_cluster_id="c1"),
        _row("rain", "g2", "c", duplicate_cluster_id="c2"),
        _row("rain", "g3", "d", duplicate_cluster_id="c3"),
    ]

    train_rows, test_rows, summary = duplicate_cluster_aware_split(rows, train_ratio=0.5, seed=3)

    train_clusters = {row["duplicate_cluster_id"] for row in train_rows}
    test_clusters = {row["duplicate_cluster_id"] for row in test_rows}
    train_groups = {row["split_group_id"] for row in train_rows}
    test_groups = {row["split_group_id"] for row in test_rows}
    assert train_clusters.isdisjoint(test_clusters)
    assert train_groups.isdisjoint(test_groups)
    assert summary["duplicate_cluster_overlap_count"] == 0
    assert summary["split_group_overlap_count"] == 0


def test_leakage_checker_reports_duplicate_cluster_overlap() -> None:
    train_rows = [_row("fog", "g0", "a", duplicate_cluster_id="dup-1")]
    test_rows = [_row("fog", "g1", "b", duplicate_cluster_id="dup-1")]

    result = check_leakage(train_rows, test_rows, dhash_threshold=0)

    assert len(result["pair_sets"]["duplicate_cluster_id"]) == 1
    assert result["coverage"]["duplicate_cluster_id_overlap_pairs"] == 1
