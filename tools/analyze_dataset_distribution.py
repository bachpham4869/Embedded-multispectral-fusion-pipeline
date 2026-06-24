#!/usr/bin/env python3
"""Analyze SmartBinocular ENV JSONL distribution for Phase 1 ML audit.

This script reads existing feature JSONL files only. It writes reproducible
distribution tables and optional class-count figures under docs/tables/ml and
docs/figures/ml by default. It does not train, tune, or overwrite any model.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import statistics
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    import numpy as np
except Exception:  # pragma: no cover - optional fallback path
    np = None  # type: ignore

try:
    from smartbinocular.feature_schema import ENV_CLASSES, FEATURE_SET_OPTICAL_ONLY
except Exception:  # pragma: no cover - keeps --help usable in broken envs
    ENV_CLASSES = [
        "night_clear",
        "normal_night",
        "normal_day",
        "fog",
        "rain",
        "glare",
        "backlight",
        "transition",
        "nir_night",
    ]
    FEATURE_SET_OPTICAL_ONLY = []


def effective_label(row: dict[str, Any]) -> str:
    """Return the effective ENV label for a JSONL row."""

    return (row.get("label") or row.get("weak_label") or "unlabeled") or "unlabeled"


def read_jsonl(paths: Iterable[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return rows


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def dataset_manifest(paths: Iterable[Path]) -> list[dict[str, Any]]:
    manifest: list[dict[str, Any]] = []
    for path in paths:
        stat = path.stat()
        manifest.append(
            {
                "path": str(path),
                "sha256": sha256_file(path),
                "bytes": stat.st_size,
            }
        )
    return manifest


def manifest_hash(manifest: list[dict[str, Any]]) -> str:
    payload = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def git_commit_hash() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip()
    except Exception:
        return "unknown"


def package_versions() -> dict[str, str]:
    versions = {
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    try:
        import sklearn

        versions["sklearn"] = sklearn.__version__
    except Exception:
        versions["sklearn"] = "unavailable"
    try:
        import numpy

        versions["numpy"] = numpy.__version__
    except Exception:
        versions["numpy"] = "unavailable"
    try:
        import scipy

        versions["scipy"] = scipy.__version__
    except Exception:
        versions["scipy"] = "unavailable"
    return versions


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return math.nan
    if np is not None:
        return float(np.percentile(np.asarray(values, dtype=float), pct))
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    pos = (len(ordered) - 1) * pct / 100.0
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(ordered[lo])
    frac = pos - lo
    return float(ordered[lo] * (1.0 - frac) + ordered[hi] * frac)


def _confidence_summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "count": 0,
            "min": None,
            "p25": None,
            "median": None,
            "mean": None,
            "p75": None,
            "max": None,
        }
    return {
        "count": len(values),
        "min": float(min(values)),
        "p25": _percentile(values, 25),
        "median": _percentile(values, 50),
        "mean": float(statistics.fmean(values)),
        "p75": _percentile(values, 75),
        "max": float(max(values)),
    }


def _recommend_class(label: str, count: int, min_class_count: int, mean_conf: float | None) -> str:
    if label == "unlabeled":
        return "exclude from classifier until labeled"
    if count < min_class_count:
        return "provisional: below support threshold; need more data or merge/drop review"
    if label == "transition":
        return "provisional: verify as ENV class vs dawn_dusk_blend runtime state"
    if label in {"glare", "backlight"}:
        return "provisional: require CI, source-diversity review, and live validation"
    if mean_conf is not None and mean_conf < 0.7:
        return "review: low mean label confidence"
    return "keep for current audit; still requires leakage and domain-shift checks"


def summarize_records(
    records: list[dict[str, Any]],
    dataset_name: str,
    min_class_count: int = 100,
) -> dict[str, Any]:
    """Summarize class/source/channel distribution for testable reuse."""

    class_counts = Counter(effective_label(row) for row in records)
    source_counts = Counter(str(row.get("source") or "unknown") for row in records)
    label_source_counts = Counter(str(row.get("label_source") or "unknown") for row in records)
    nir_channel_counts = Counter(str(row.get("nir_channel") or "unknown") for row in records)
    thermal_channel_counts = Counter(str(row.get("thermal_channel") or "unknown") for row in records)

    by_class_source: dict[str, Counter[str]] = defaultdict(Counter)
    confidence_by_class: dict[str, list[float]] = defaultdict(list)
    confidence_values: list[float] = []
    for row in records:
        label = effective_label(row)
        source = str(row.get("source") or "unknown")
        by_class_source[label][source] += 1
        conf_raw = row.get("label_confidence")
        if conf_raw is None:
            continue
        try:
            conf = float(conf_raw)
        except (TypeError, ValueError):
            continue
        confidence_values.append(conf)
        confidence_by_class[label].append(conf)

    nonzero_counts = [v for v in class_counts.values() if v > 0]
    imbalance_ratio = (
        float(max(nonzero_counts) / min(nonzero_counts)) if nonzero_counts else math.nan
    )
    row_count = len(records)

    class_percentages = {
        label: (count / row_count * 100.0 if row_count else 0.0)
        for label, count in class_counts.items()
    }
    low_support = sorted(
        [label for label, count in class_counts.items() if count < min_class_count]
    )

    class_rows: list[dict[str, Any]] = []
    for label in sorted(class_counts.keys(), key=lambda x: (ENV_CLASSES.index(x) if x in ENV_CLASSES else 99, x)):
        count = class_counts[label]
        conf = _confidence_summary(confidence_by_class.get(label, []))
        mean_conf = conf["mean"] if isinstance(conf["mean"], float) else None
        top_sources = by_class_source[label].most_common(5)
        class_rows.append(
            {
                "class": label,
                "count": count,
                "percentage": class_percentages[label],
                "mean_label_confidence": mean_conf,
                "top_sources": top_sources,
                "recommendation": _recommend_class(label, count, min_class_count, mean_conf),
            }
        )

    return {
        "dataset_name": dataset_name,
        "row_count": row_count,
        "class_counts": dict(class_counts),
        "class_percentages": class_percentages,
        "source_counts": dict(source_counts),
        "label_source_counts": dict(label_source_counts),
        "nir_channel_counts": dict(nir_channel_counts),
        "thermal_channel_counts": dict(thermal_channel_counts),
        "imbalance_ratio": imbalance_ratio,
        "low_support_classes": low_support,
        "label_confidence": _confidence_summary(confidence_values),
        "class_rows": class_rows,
        "by_class_source": {k: dict(v) for k, v in by_class_source.items()},
    }


def _fmt_float(value: Any, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    try:
        x = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(x):
        return "n/a"
    return f"{x:.{digits}f}"


def _md_table(headers: list[str], rows: list[list[Any]]) -> str:
    out = ["| " + " | ".join(headers) + " |"]
    out.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        out.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(out) + "\n"


def _write_dataset_distribution(path: Path, summary: dict[str, Any], provenance: dict[str, Any]) -> None:
    rows: list[list[Any]] = []
    for item in summary["class_rows"]:
        rows.append(
            [
                item["class"],
                item["count"],
                _fmt_float(item["percentage"], 2),
                _fmt_float(item["mean_label_confidence"], 3),
                ", ".join(f"{src}:{count}" for src, count in item["top_sources"]) or "n/a",
                item["recommendation"],
            ]
        )

    conf = summary["label_confidence"]
    body = [
        f"# Dataset Distribution: {summary['dataset_name']}",
        "",
        "Status: distribution evidence. Classification metrics are not included here.",
        "",
        "## Reproducibility",
        "",
        _md_table(
            ["Field", "Value"],
            [
                ["command", provenance["command"]],
                ["git_commit", provenance["git_commit"]],
                ["dataset_manifest_hash", provenance["manifest_hash"]],
                ["row_count", summary["row_count"]],
                ["feature_set_version", provenance["feature_set_version"]],
                ["python", provenance["versions"]["python"]],
                ["sklearn", provenance["versions"]["sklearn"]],
                ["numpy", provenance["versions"]["numpy"]],
                ["scipy", provenance["versions"]["scipy"]],
            ],
        ),
        "Dataset files:",
        "",
        _md_table(
            ["Path", "SHA256", "Bytes"],
            [[m["path"], m["sha256"], m["bytes"]] for m in provenance["manifest"]],
        ),
        "## Per-Class Distribution",
        "",
        _md_table(
            ["Class", "Count", "Percent", "Mean Confidence", "Top Sources", "Recommendation"],
            rows,
        ),
        "## Label and Channel Summary",
        "",
        _md_table(
            ["Category", "Distribution"],
            [
                ["label_source", ", ".join(f"{k}:{v}" for k, v in summary["label_source_counts"].items())],
                ["nir_channel", ", ".join(f"{k}:{v}" for k, v in summary["nir_channel_counts"].items())],
                [
                    "thermal_channel",
                    ", ".join(f"{k}:{v}" for k, v in summary["thermal_channel_counts"].items()),
                ],
                ["imbalance_ratio_max_min", _fmt_float(summary["imbalance_ratio"], 3)],
                ["low_support_classes", ", ".join(summary["low_support_classes"]) or "none"],
            ],
        ),
        "## Label Confidence",
        "",
        _md_table(
            ["Count", "Min", "P25", "Median", "Mean", "P75", "Max"],
            [
                [
                    conf["count"],
                    _fmt_float(conf["min"], 3),
                    _fmt_float(conf["p25"], 3),
                    _fmt_float(conf["median"], 3),
                    _fmt_float(conf["mean"], 3),
                    _fmt_float(conf["p75"], 3),
                    _fmt_float(conf["max"], 3),
                ]
            ],
        ),
    ]
    path.write_text("\n".join(body), encoding="utf-8")


def _write_source_distribution(path: Path, summaries: list[dict[str, Any]], provenance: dict[str, Any]) -> None:
    rows: list[list[Any]] = []
    for summary in summaries:
        total = max(1, int(summary["row_count"]))
        for source, count in sorted(summary["source_counts"].items()):
            rows.append(
                [
                    summary["dataset_name"],
                    source,
                    count,
                    _fmt_float(count / total * 100.0, 2),
                ]
            )
    body = [
        "# Source Distribution",
        "",
        "Status: source-level distribution evidence. Source overlap is not leakage by itself, but it must be reviewed with the leakage protocol.",
        "",
        f"Command: `{provenance['command']}`",
        f"Git commit: `{provenance['git_commit']}`",
        "",
        _md_table(["Dataset", "Source", "Count", "Percent"], rows),
    ]
    path.write_text("\n".join(body), encoding="utf-8")


def _write_imbalance_summary(path: Path, summaries: list[dict[str, Any]], provenance: dict[str, Any]) -> None:
    rows = [
        [
            summary["dataset_name"],
            summary["row_count"],
            _fmt_float(summary["imbalance_ratio"], 3),
            ", ".join(summary["low_support_classes"]) or "none",
        ]
        for summary in summaries
    ]
    body = [
        "# Class Imbalance Summary",
        "",
        f"Command: `{provenance['command']}`",
        f"Git commit: `{provenance['git_commit']}`",
        "",
        _md_table(["Dataset", "Rows", "Max/Min Class Ratio", "Below Threshold"], rows),
    ]
    path.write_text("\n".join(body), encoding="utf-8")


def _write_class_support_risk(path: Path, summaries: list[dict[str, Any]], provenance: dict[str, Any]) -> None:
    by_class: dict[str, dict[str, Any]] = {}
    for summary in summaries:
        name = summary["dataset_name"]
        for item in summary["class_rows"]:
            entry = by_class.setdefault(item["class"], {"class": item["class"]})
            entry[f"{name}_count"] = item["count"]
            entry[f"{name}_pct"] = item["percentage"]
            entry[f"{name}_mean_conf"] = item["mean_label_confidence"]
            entry[f"{name}_recommendation"] = item["recommendation"]

    dataset_names = [s["dataset_name"] for s in summaries]
    headers = ["Class"]
    for name in dataset_names:
        headers.extend([f"{name} Count", f"{name} Percent", f"{name} Mean Conf"])
    headers.extend(["Phase 1 Risk", "Recommendation"])

    rows: list[list[Any]] = []
    for label in sorted(by_class.keys(), key=lambda x: (ENV_CLASSES.index(x) if x in ENV_CLASSES else 99, x)):
        entry = by_class[label]
        row: list[Any] = [label]
        for name in dataset_names:
            row.extend(
                [
                    entry.get(f"{name}_count", 0),
                    _fmt_float(entry.get(f"{name}_pct", 0.0), 2),
                    _fmt_float(entry.get(f"{name}_mean_conf"), 3),
                ]
            )
        if label == "transition":
            risk = "high taxonomy risk"
            rec = "Do not claim strong ENV class until decision record evidence is complete."
        elif label in {"glare", "backlight"}:
            risk = "high support/source-diversity risk"
            rec = "Keep provisional; require CI, source diversity, and live validation."
        elif any("provisional" in str(entry.get(f"{name}_recommendation", "")) for name in dataset_names):
            risk = "support risk"
            rec = "Need more data or merge/drop review before strong thesis claim."
        else:
            risk = "standard"
            rec = "Usable for current audit subject to leakage/domain-shift checks."
        row.extend([risk, rec])
        rows.append(row)

    body = [
        "# Class Support Risk Summary",
        "",
        "Status: Phase 1 risk table. It does not prove classifier quality.",
        "",
        f"Command: `{provenance['command']}`",
        f"Git commit: `{provenance['git_commit']}`",
        "",
        _md_table(headers, rows),
    ]
    path.write_text("\n".join(body), encoding="utf-8")


def _write_csv(path: Path, headers: list[str], rows: list[list[Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)


def _write_distribution_csv(path: Path, summary: dict[str, Any]) -> None:
    rows = [
        [
            item["class"],
            item["count"],
            f"{item['percentage']:.6f}",
            "" if item["mean_label_confidence"] is None else f"{item['mean_label_confidence']:.6f}",
            ";".join(f"{src}:{count}" for src, count in item["top_sources"]),
            item["recommendation"],
        ]
        for item in summary["class_rows"]
    ]
    _write_csv(
        path,
        ["class", "count", "percentage", "mean_label_confidence", "top_sources", "recommendation"],
        rows,
    )


def _plot_class_distribution(path: Path, summary: dict[str, Any]) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return False

    labels = [item["class"] for item in summary["class_rows"]]
    counts = [item["count"] for item in summary["class_rows"]]
    fig_width = max(8, len(labels) * 0.8)
    fig, ax = plt.subplots(figsize=(fig_width, 4.8))
    bars = ax.bar(labels, counts, color="#2d6a9f")
    ax.set_title(f"Class distribution: {summary['dataset_name']}")
    ax.set_xlabel("ENV class")
    ax.set_ylabel("Rows")
    ax.tick_params(axis="x", rotation=35)
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), str(count), ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return True


def _make_provenance(paths: list[Path], command: str) -> dict[str, Any]:
    manifest = dataset_manifest(paths)
    return {
        "command": command,
        "git_commit": git_commit_hash(),
        "manifest": manifest,
        "manifest_hash": manifest_hash(manifest),
        "feature_set_version": f"optical_12_baseline ({len(FEATURE_SET_OPTICAL_ONLY)} features)",
        "versions": package_versions(),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Analyze existing SmartBinocular ENV JSONL class/source/channel distribution."
    )
    p.add_argument("--train", type=Path, nargs="+", required=True, help="Training JSONL file(s)")
    p.add_argument("--test", type=Path, nargs="*", default=[], help="Optional held-out test JSONL file(s)")
    p.add_argument(
        "--reference",
        type=Path,
        nargs="*",
        default=[],
        help="Optional full/reference JSONL file(s), e.g. merged_logs_ml.jsonl",
    )
    p.add_argument("--out-dir", type=Path, default=Path("docs/tables/ml"), help="Markdown/CSV output directory")
    p.add_argument("--fig-dir", type=Path, default=Path("docs/figures/ml"), help="Figure output directory")
    p.add_argument("--min-train-count", type=int, default=500, help="Minimum train support per class")
    p.add_argument("--min-test-count", type=int, default=100, help="Minimum test support per class")
    p.add_argument("--no-plots", action="store_true", help="Skip PNG figure generation")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    for path in [*args.train, *args.test, *args.reference]:
        if not path.is_file():
            print(f"ERROR: JSONL not found: {path}", file=sys.stderr)
            return 2

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.fig_dir.mkdir(parents=True, exist_ok=True)
    command = " ".join([sys.executable, *sys.argv])

    generated: list[Path] = []
    summaries: list[dict[str, Any]] = []

    train_rows = read_jsonl(args.train)
    train_summary = summarize_records(train_rows, "train", args.min_train_count)
    train_prov = _make_provenance(args.train, command)
    _write_dataset_distribution(args.out_dir / "dataset_distribution_train.md", train_summary, train_prov)
    _write_distribution_csv(args.out_dir / "dataset_distribution_train.csv", train_summary)
    generated.extend([args.out_dir / "dataset_distribution_train.md", args.out_dir / "dataset_distribution_train.csv"])
    summaries.append(train_summary)

    if args.test:
        test_rows = read_jsonl(args.test)
        test_summary = summarize_records(test_rows, "test", args.min_test_count)
        test_prov = _make_provenance(args.test, command)
        _write_dataset_distribution(args.out_dir / "dataset_distribution_test.md", test_summary, test_prov)
        _write_distribution_csv(args.out_dir / "dataset_distribution_test.csv", test_summary)
        generated.extend([args.out_dir / "dataset_distribution_test.md", args.out_dir / "dataset_distribution_test.csv"])
        summaries.append(test_summary)

    if args.reference:
        reference_rows = read_jsonl(args.reference)
        reference_summary = summarize_records(reference_rows, "reference", args.min_train_count)
        reference_prov = _make_provenance(args.reference, command)
        _write_dataset_distribution(args.out_dir / "dataset_distribution_reference.md", reference_summary, reference_prov)
        _write_distribution_csv(args.out_dir / "dataset_distribution_reference.csv", reference_summary)
        generated.extend([args.out_dir / "dataset_distribution_reference.md", args.out_dir / "dataset_distribution_reference.csv"])
        summaries.append(reference_summary)

    all_paths = [*args.train, *args.test, *args.reference]
    shared_prov = _make_provenance(all_paths, command)
    _write_source_distribution(args.out_dir / "source_distribution.md", summaries, shared_prov)
    _write_imbalance_summary(args.out_dir / "class_imbalance_summary.md", summaries, shared_prov)
    _write_class_support_risk(args.out_dir / "class_support_risk_summary.md", summaries, shared_prov)
    generated.extend(
        [
            args.out_dir / "source_distribution.md",
            args.out_dir / "class_imbalance_summary.md",
            args.out_dir / "class_support_risk_summary.md",
        ]
    )

    if not args.no_plots:
        if _plot_class_distribution(args.fig_dir / "class_distribution_train.png", train_summary):
            generated.append(args.fig_dir / "class_distribution_train.png")
        if args.test and _plot_class_distribution(args.fig_dir / "class_distribution_test.png", summaries[1]):
            generated.append(args.fig_dir / "class_distribution_test.png")

    for path in generated:
        print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
