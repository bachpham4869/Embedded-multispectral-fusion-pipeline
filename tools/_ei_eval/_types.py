"""Shared types re-exported to avoid circular imports between discover and metrics."""

from .discover import GTBox, EvalItem

__all__ = ["GTBox", "EvalItem"]
