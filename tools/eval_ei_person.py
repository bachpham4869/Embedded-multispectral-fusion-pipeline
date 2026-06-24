"""Offline batch evaluator for the Edge Impulse FOMO person-in-dark model.

Canonical baseline (locked, see DECISIONS_AND_RISKS.md Q1–Q5):
    python tools/eval_ei_person.py \\
      --tflite models/ei/person_in_dark_fomo_int8.tflite \\
      --dataset data/find-person-in-the-dark/train/train \\
      --pass raw --threshold 0.8 --fit-mode crop --interp area \\
      --limit 500 --metric-primary centroid_hit \\
      --run-id baseline_train500_crop_t080

Producer constraint: this script does NOT modify any live pipeline code or config defaults.
The ≤50 ms/frame producer budget for python -m smartbinocular is unaffected.
"""

from __future__ import annotations

import argparse
import datetime
import sys
from pathlib import Path

# Ensure src/ package is importable when run directly from repo root
_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
# Also add tools/ itself for the _ei_eval package
_TOOLS = _REPO_ROOT / "tools"
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))


def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Offline EI person-in-dark evaluation harness (train-only by default).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--tflite",
        required=True,
        metavar="PATH",
        help="Path to the .tflite model file.",
    )
    p.add_argument(
        "--dataset",
        required=True,
        metavar="PATH",
        help="Path to the train/train directory (contains train_images/ and train_annotations/).",
    )
    p.add_argument(
        "--pass",
        dest="pass_mode",
        choices=["raw", "pipeline_lite"],
        default="raw",
        help="Preprocessing pass. 'raw' uses the production _prepare_ei_input; "
             "'pipeline_lite' applies the pipeline-lite variant first (v0=identity).",
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=0.8,
        metavar="FLOAT",
        help="Detection score threshold (default: 0.8, matching production config).",
    )
    p.add_argument(
        "--fit-mode",
        dest="fit_mode",
        choices=["crop", "letterbox", "passthrough"],
        default="crop",
        help="Resize/crop policy. 'crop' is the canonical baseline matching "
             "EI_CLASSIFIER_RESIZE_FIT_SHORTEST (see DECISIONS_AND_RISKS.md Q4).",
    )
    p.add_argument(
        "--interp",
        choices=["area", "linear", "nearest"],
        default="area",
        help="OpenCV interpolation flag for resize (default: area).",
    )
    p.add_argument(
        "--pipeline-lite-variant",
        dest="pipeline_lite_variant",
        choices=["identity", "clahe", "gamma"],
        default="identity",
        help="Pipeline-lite preprocessing variant. v0 only supports 'identity'.",
    )
    p.add_argument(
        "--metric-primary",
        dest="metric_primary",
        choices=["centroid_hit", "image_f1"],
        default="centroid_hit",
        help="Primary tuning metric (default: centroid_hit per Q1).",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=500,
        metavar="N",
        help="Evaluate at most N images (default: 500). Use --limit 0 only with --allow-full-run.",
    )
    p.add_argument(
        "--allow-full-run",
        dest="allow_full_run",
        action="store_true",
        default=False,
        help="Required to pass --limit 0 (full dataset). Not enabled in v0.",
    )
    p.add_argument(
        "--sanity-test-limit",
        dest="sanity_test_limit",
        type=int,
        default=0,
        metavar="N",
        help="If > 0, run an additional pass on the first N images of test/test/test_images/ "
             "and write to <out>/<run_id>/sanity/. Off by default.",
    )
    p.add_argument(
        "--run-id",
        dest="run_id",
        default=None,
        metavar="ID",
        help="Run identifier (default: ISO timestamp).",
    )
    p.add_argument(
        "--out",
        default=None,
        metavar="PATH",
        help="Output root directory (default: docs/eval/ei_person_find_in_dark/<run_id>/).",
    )
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)

    if args.limit == 0 and not args.allow_full_run:
        print(
            "ERROR: --limit 0 requires --allow-full-run. "
            "Full-dataset runs are not enabled in v0 (see DECISIONS_AND_RISKS.md).",
            file=sys.stderr,
        )
        return 2

    run_id = args.run_id or datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
    epoch_name = (
        f"epoch_00_{args.pass_mode}_t{int(args.threshold * 100):03d}"
        f"_{args.fit_mode}_{args.interp}"
    )

    if args.out:
        out_root = Path(args.out)
    else:
        out_root = (
            Path(__file__).resolve().parent.parent
            / "docs" / "eval" / "ei_person_find_in_dark"
            / run_id
        )
    epoch_dir = out_root / epoch_name

    dataset_root = Path(args.dataset)
    tflite_path = args.tflite

    # Sanity-test dataset lives one level up (data/find-person-in-the-dark/test/test)
    sanity_root = None
    if args.sanity_test_limit > 0:
        candidate = dataset_root.parent.parent / "test" / "test"
        if candidate.is_dir():
            sanity_root = candidate
        else:
            print(
                f"WARNING: --sanity-test-limit {args.sanity_test_limit} requested but "
                f"test split not found at {candidate} — skipping sanity pass.",
                file=sys.stderr,
            )

    from _ei_eval.runner import run_epoch

    try:
        summary = run_epoch(
            tflite_path=tflite_path,
            dataset_root=dataset_root,
            out_dir=epoch_dir,
            threshold=args.threshold,
            fit_mode=args.fit_mode,
            interp=args.interp,
            pipeline_lite_variant=args.pipeline_lite_variant,
            metric_primary=args.metric_primary,
            limit=args.limit,
            run_id=run_id,
            epoch_name=epoch_name,
            argv=sys.argv[:],
            sanity_dataset_root=sanity_root,
            sanity_limit=args.sanity_test_limit,
        )
    except (ValueError, FileNotFoundError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(f"\nResults written to: {epoch_dir}")
    print(f"  summary.json    — all metrics")
    print(f"  per_image.csv   — per-image breakdown")
    print(f"  params.yaml     — reproducibility record")
    print(f"  padilla_gt/     — derived GT for secondary mAP (pascalvoc.py)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
