#!/usr/bin/env python3
"""Audit local options for raw-sensor labeling support.

This tool does not download models or call external APIs. It documents whether
an independent teacher is locally available; when none is available, the only
enabled path is RF/heuristic `suggested_label` support for manual review.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from smartbinocular.feature_schema import ENV_CLASSES
from tools.ml_metadata_utils import git_branch_name, git_commit_hash, markdown_table, package_versions, write_text


PROMPT_MAPPING = {
    "night_clear": "clear outdoor night scene with very low ambient light",
    "normal_night": "low-light night scene with ambient street or urban lighting",
    "normal_day": "normal daylight outdoor scene",
    "fog": "foggy or hazy low-visibility outdoor scene",
    "rain": "rainy outdoor scene, wet road, wet lens, or visible rain streaks",
    "glare": "direct glare from bright light source such as sun or headlights",
    "backlight": "backlit scene with subject or foreground against strong background light",
    "nir_night": "active infrared or NIR monochrome night scene",
    "transition": "dawn or dusk transition lighting, not just model uncertainty",
    "unknown_or_out_of_scope": "uncertain, ambiguous, indoor, non-environmental, or outside the taxonomy",
}


def _available(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


def _env_present(names: list[str]) -> bool:
    return any(bool(os.environ.get(name)) for name in names)


def audit_labeling_support_options() -> dict[str, Any]:
    clip_available = _available("clip") or _available("open_clip")
    torchvision_available = _available("torchvision")
    timm_available = _available("timm")
    onnx_available = _available("onnxruntime")
    tflite_available = _available("tflite_runtime") or _available("tensorflow")
    api_configured = _env_present(["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY"])
    independent_teacher_available = any([clip_available, torchvision_available, timm_available, onnx_available, tflite_available, api_configured])

    candidates = [
        {
            "candidate_method": "clip_zero_shot",
            "input_required": "sampled RGB/optical frames + class prompts",
            "local_availability": "available" if clip_available else "unavailable",
            "label_mapping_feasibility": "medium if local CLIP/OpenCLIP weights exist",
            "expected_reliability": "independent weak teacher only after validation",
            "compute_cost": "medium/high on CPU",
            "limitations": "not installed locally in current audit; no downloads in this phase",
            "output_label_type": "auto_weak_label only if enabled in a separate approved plan",
            "selected_reason": "deferred; independent teacher unavailable locally" if not clip_available else "available but requires separate teacher-labeling plan",
        },
        {
            "candidate_method": "torchvision_pretrained",
            "input_required": "sampled frames",
            "local_availability": "available" if torchvision_available else "unavailable",
            "label_mapping_feasibility": "low/medium; ImageNet labels do not map directly to ENV classes",
            "expected_reliability": "limited without weather/night fine-tuning",
            "compute_cost": "medium on CPU",
            "limitations": "no direct ENV taxonomy output; no downloads in this phase",
            "output_label_type": "auto_weak_label only if a local model is approved",
            "selected_reason": "deferred; no local torchvision stack" if not torchvision_available else "not selected as same-phase teacher",
        },
        {
            "candidate_method": "timm_pretrained",
            "input_required": "sampled frames",
            "local_availability": "available" if timm_available else "unavailable",
            "label_mapping_feasibility": "medium if Places/weather model weights are local",
            "expected_reliability": "unknown until validated",
            "compute_cost": "medium/high on CPU",
            "limitations": "not installed locally in current audit; no downloads in this phase",
            "output_label_type": "auto_weak_label only after separate approval",
            "selected_reason": "deferred; no local timm stack" if not timm_available else "requires separate teacher-labeling plan",
        },
        {
            "candidate_method": "onnx_tflite_local_classifier",
            "input_required": "sampled frames + local ONNX/TFLite model",
            "local_availability": "available" if (onnx_available or tflite_available) else "unavailable",
            "label_mapping_feasibility": "depends on model labels",
            "expected_reliability": "unknown until model and mapping are audited",
            "compute_cost": "low/medium if edge model is small",
            "limitations": "no local runtime/model found in current audit",
            "output_label_type": "auto_weak_label only after model provenance is documented",
            "selected_reason": "deferred; no local runtime/model available" if not (onnx_available or tflite_available) else "requires separate teacher-labeling plan",
        },
        {
            "candidate_method": "external_vlm_api",
            "input_required": "sampled frames + API credentials + user approval",
            "local_availability": "configured" if api_configured else "not configured",
            "label_mapping_feasibility": "high with prompt mapping, but external and paid/remote",
            "expected_reliability": "potential independent review aid, not ground truth",
            "compute_cost": "external API cost; latency depends on provider",
            "limitations": "not called without explicit approval",
            "output_label_type": "auto_weak_label only after separate approved protocol",
            "selected_reason": "deferred; user approval required",
        },
        {
            "candidate_method": "rf_heuristic_suggested_label",
            "input_required": "production RF predictions + optical features + domain-shift metrics",
            "local_availability": "available",
            "label_mapping_feasibility": "high for suggestions because it already emits ENV labels",
            "expected_reliability": "not independent; only review-prioritization support",
            "compute_cost": "low",
            "limitations": "RF and heuristics share feature signals; agreement is not independent evidence",
            "output_label_type": "suggested_label",
            "selected_reason": "selected for manual-review acceleration; not an auto weak label source",
        },
    ]

    return {
        "status": "no_independent_teacher_available" if not independent_teacher_available else "independent_teacher_candidate_available",
        "independent_teacher_available": independent_teacher_available,
        "selected_method": "rf_heuristic_suggested_label",
        "weak_label_dataset_allowed": False,
        "candidates": candidates,
        "prompt_mapping": PROMPT_MAPPING,
        "git_commit": git_commit_hash(),
        "git_branch": git_branch_name(),
        "package_versions": package_versions(),
    }


def write_candidates_table(path: Path, result: dict[str, Any]) -> None:
    rows = [
        [
            c["candidate_method"],
            c["local_availability"],
            c["output_label_type"],
            c["label_mapping_feasibility"],
            c["compute_cost"],
            c["limitations"],
            c["selected_reason"],
        ]
        for c in result["candidates"]
    ]
    write_text(
        path,
        "\n".join(
            [
                "# Auto-Labeling Model Candidates",
                "",
                "Status: no independent teacher is used in the current phase. RF/heuristic output is `suggested_label` only.",
                "",
                markdown_table(
                    ["Method", "Availability", "Label type", "Mapping", "Cost", "Limitations", "Decision"],
                    rows,
                ),
            ]
        )
        + "\n",
    )


def write_options_doc(path: Path, result: dict[str, Any]) -> None:
    write_text(
        path,
        "\n".join(
            [
                "# Auto-Labeling Options",
                "",
                "Current decision: do not generate `auto_weak_label` records in this phase because no local independent teacher is available and no external API/model download has been approved.",
                "",
                markdown_table(
                    ["Field", "Value"],
                    [
                        ["status", result["status"]],
                        ["selected_method", result["selected_method"]],
                        ["independent_teacher_available", result["independent_teacher_available"]],
                        ["weak_label_dataset_allowed", result["weak_label_dataset_allowed"]],
                        ["git_commit", result["git_commit"]],
                    ],
                ),
                "",
                "RF/heuristic suggestions are not a stronger model and not an independent teacher. They only pre-fill a review template and prioritize frames for human labeling.",
            ]
        )
        + "\n",
    )


def write_prompt_mapping(path: Path, result: dict[str, Any]) -> None:
    rows = []
    for label in list(ENV_CLASSES) + ["unknown_or_out_of_scope"]:
        rows.append([label, result["prompt_mapping"][label], _prompt_caveat(label)])
    write_text(
        path,
        "\n".join(
            [
                "# Auto-Label Class Prompt Mapping",
                "",
                "These prompts are for a future independent teacher path. In the current phase, they are documentation only.",
                "",
                markdown_table(["ENV class", "Prompt", "Caveat"], rows),
            ]
        )
        + "\n",
    )


def _prompt_caveat(label: str) -> str:
    if label == "transition":
        return "Use only for true dawn/dusk transient lighting, not uncertainty."
    if label == "nir_night":
        return "Low confidence unless modality is confirmed NIR/IR-like."
    if label == "unknown_or_out_of_scope":
        return "Allowed when no class is reliable."
    return "Standard taxonomy mapping."


def write_protocol(path: Path, result: dict[str, Any]) -> None:
    write_text(
        path,
        "\n".join(
            [
                "# Auto-Labeling Protocol",
                "",
                "Label layers are intentionally separated:",
                "",
                "- `suggested_label`: RF/heuristic review aid. It is not ground truth and not an independent teacher.",
                "- `auto_weak_label`: reserved for a future independent teacher such as CLIP/VLM/local pretrained model after user approval.",
                "- `manual_label`: the only ground truth for raw sensor accuracy.",
                "",
                "Current phase output is limited to `suggested_label`, review priority, and consistency analysis. Agreement between RF and heuristic is not independent confidence because both use related optical feature signals.",
                "",
                "Before any `auto_weak_label` dataset is generated, create a separate plan documenting teacher model provenance, prompt mapping, confidence thresholds, cost, and human verification workflow.",
            ]
        )
        + "\n",
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit local options for raw-sensor labeling support.")
    parser.add_argument("--options-md", type=Path, required=True)
    parser.add_argument("--candidates-md", type=Path, required=True)
    parser.add_argument("--protocol-md", type=Path, required=True)
    parser.add_argument("--prompt-md", type=Path, required=True)
    parser.add_argument("--out-json", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = audit_labeling_support_options()
    write_options_doc(args.options_md, result)
    write_candidates_table(args.candidates_md, result)
    write_protocol(args.protocol_md, result)
    write_prompt_mapping(args.prompt_md, result)
    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote {args.options_md}")
    print(f"Wrote {args.candidates_md}")
    print(f"Wrote {args.protocol_md}")
    print(f"Wrote {args.prompt_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
