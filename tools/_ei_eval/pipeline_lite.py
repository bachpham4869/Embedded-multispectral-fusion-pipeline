"""Pipeline-lite v0: identity-only preprocessing variants for offline eval.

v0 scope: {crop, letterbox, passthrough} × {area, linear, nearest}.
No contrast bump, no gamma, no CLAHE (see DECISIONS_AND_RISKS.md Q2).

Contrast variants are present as enum values but raise NotImplementedError
until the Q2 trigger fires (centroid-hit recall plateau across all fit_mode×interp
permutations AND gap between hit-rate and image-F1 ≥ 0.05).
"""

from __future__ import annotations

import enum


class PipelineLiteVariant(str, enum.Enum):
    IDENTITY = "identity"
    CLAHE = "clahe"        # future: raises NotImplementedError in v0
    GAMMA = "gamma"        # future: raises NotImplementedError in v0


_SUPPORTED_FIT_MODES = frozenset({"crop", "letterbox", "passthrough"})
_SUPPORTED_INTERPS = frozenset({"area", "linear", "nearest"})


def apply(
    bgr,
    *,
    variant: PipelineLiteVariant = PipelineLiteVariant.IDENTITY,
    fit_mode: str = "crop",
    interp: str = "area",
):
    """Apply pipeline-lite preprocessing and return the frame unchanged.

    In v0, all variants are identity (no pixel transformation).  The fit_mode
    and interp are validated here but applied inside runtime.py's _prepare_ei_input.
    This function returns bgr as-is; it exists so the runner can call a consistent
    API when future contrast variants are enabled.
    """
    if variant != PipelineLiteVariant.IDENTITY:
        raise NotImplementedError(
            f"pipeline_lite variant '{variant.value}' is not enabled in v0. "
            "See DECISIONS_AND_RISKS.md Q2 for the trigger condition."
        )
    if fit_mode not in _SUPPORTED_FIT_MODES:
        raise ValueError(f"fit_mode must be one of {sorted(_SUPPORTED_FIT_MODES)}, got {fit_mode!r}")
    if interp not in _SUPPORTED_INTERPS:
        raise ValueError(f"interp must be one of {sorted(_SUPPORTED_INTERPS)}, got {interp!r}")
    return bgr
