#!/usr/bin/env python3
"""Raspberry Pi field helpers for thesis-grade artefacts (timing JSON, host telemetry).

Run on the device **beside** ``python -m smartbinocular``. Typical workflow::

    # Terminal A --- sampled throttle / CPU / RAM while the pipeline runs
    python tools/rpi_field_collect.py host-monitor --duration_sec 3600 --interval_sec 5

    # Terminal B --- interactive fusion session (stop with q); writes session_*.json under fusion_captures/metrics/
    ML_INFERENCE_ENABLED=1 env_mode=auto_rule python -m smartbinocular

After stopping the pipeline, bundle recent metrics for rsync to dev::

    python tools/rpi_field_collect.py bundle-recent --max_sessions 8

Outputs live under ``<fusion_root>/field_runs/`` (default: ``./fusion_captures`` next to cwd,
or ``~/fusion_captures`` when the repo cwd is not writable --- same rule as the pipeline).

See also: ``rsync.sh`` pull targets on the development machine.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _try_import_throttle_snapshot():
    try:
        from smartbinocular.metrics import _capture_throttle_snapshot  # noqa: SLF001

        return _capture_throttle_snapshot
    except Exception:
        return None


def _vcgencmd_snapshot() -> Dict[str, Any]:
    """Fallback when smartbinocular is not importable; Pi-only."""
    snap: Dict[str, Any] = {
        "vcgencmd_throttled": None,
        "vcgencmd_temp": None,
        "cpu_freq_khz": None,
        "platform": platform.machine(),
    }
    vc = shutil.which("vcgencmd") or "/usr/bin/vcgencmd"
    try:
        snap["vcgencmd_throttled"] = subprocess.check_output(
            [vc, "get_throttled"], timeout=2, stderr=subprocess.DEVNULL
        ).decode()
        snap["vcgencmd_temp"] = subprocess.check_output(
            [vc, "measure_temp"], timeout=2, stderr=subprocess.DEVNULL
        ).decode()
    except (FileNotFoundError, subprocess.CalledProcessError, OSError):
        pass
    freqs: List[int] = []
    try:
        for p in sorted(Path("/sys/devices/system/cpu").glob("cpu*/cpufreq/scaling_cur_freq")):
            freqs.append(int(p.read_text().strip()))
    except (OSError, ValueError):
        freqs = []
    snap["cpu_freq_khz"] = freqs or None
    return snap


def _mem_kb() -> Tuple[Optional[int], Optional[int]]:
    total = avail = None
    try:
        with open("/proc/meminfo", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("MemTotal:"):
                    total = int(line.split()[1])
                elif line.startswith("MemAvailable:"):
                    avail = int(line.split()[1])
    except OSError:
        pass
    return total, avail


def _loadavg() -> Optional[Tuple[float, float, float]]:
    try:
        parts = Path("/proc/loadavg").read_text(encoding="utf-8").split()
        return float(parts[0]), float(parts[1]), float(parts[2])
    except (OSError, IndexError, ValueError):
        return None


def resolve_fusion_root(explicit: Optional[Path]) -> Path:
    """Mirror ``ensure_fusion_capture_dirs`` preference: cwd first, then home."""
    if explicit is not None:
        return explicit.expanduser().resolve()
    cwd_candidate = Path.cwd() / "fusion_captures"
    home_candidate = Path.home() / "fusion_captures"
    for p in (cwd_candidate, home_candidate):
        if p.is_dir():
            return p.resolve()
    # Prefer creating next to cwd (caller may mkdir via pipeline)
    return cwd_candidate.resolve()


def resolve_metrics_dir(fusion_root: Path) -> Path:
    return (fusion_root / "metrics").resolve()


def _git_revision(repo: Path) -> Optional[str]:
    try:
        out = subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
        return out.decode().strip()
    except (FileNotFoundError, subprocess.CalledProcessError, OSError):
        return None


def cmd_run_note(args: argparse.Namespace) -> int:
    fusion_root = resolve_fusion_root(Path(args.fusion_root) if args.fusion_root else None)
    notes_dir = fusion_root / "field_runs" / "run_notes"
    notes_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe_tag = "".join(c if c.isalnum() or c in "-_" else "_" for c in args.tag)[:80]
    path = notes_dir / f"run_{stamp}_{safe_tag}.json"
    env_keys = [k.strip() for k in args.capture_env.split(",") if k.strip()]
    env_snap = {k: os.environ.get(k) for k in env_keys}
    repo = _repo_root()
    payload = {
        "schema": "smartbinocular_field_run_note_v1",
        "utc_created": stamp,
        "tag": args.tag,
        "notes": args.notes,
        "hostname": platform.node(),
        "machine": platform.machine(),
        "git_revision": _git_revision(repo),
        "cwd": str(Path.cwd()),
        "fusion_root": str(fusion_root),
        "pipeline_hints": {
            "imx_hotkey": "1",
            "thermal_hotkey": "2",
            "fusion_hotkey": "3",
            "quit_hotkey": "q",
            "remark": "CLI --mode was removed; use numeric keys after startup.",
        },
        "captured_env": env_snap,
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {path}")
    return 0


def cmd_host_monitor(args: argparse.Namespace) -> int:
    fusion_root = resolve_fusion_root(Path(args.fusion_root) if args.fusion_root else None)
    out_dir = fusion_root / "field_runs" / "monitors"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = out_dir / f"host_{stamp}.ndjson"

    throttle_fn = _try_import_throttle_snapshot()
    interval = max(0.5, float(args.interval_sec))
    duration = float(args.duration_sec)
    t_end = time.perf_counter() + duration

    print(f"Writing NDJSON samples every {interval}s for {duration}s -> {out_path}")
    with out_path.open("w", encoding="utf-8") as fh:
        while time.perf_counter() < t_end:
            ts = time.time()
            mem_total, mem_avail = _mem_kb()
            lav = _loadavg()
            if throttle_fn is not None:
                try:
                    th = throttle_fn()
                except Exception:
                    th = _vcgencmd_snapshot()
            else:
                th = _vcgencmd_snapshot()
            row = {
                "unix_ts": ts,
                "utc_iso": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "loadavg_1_5_15": lav,
                "mem_total_kb": mem_total,
                "mem_avail_kb": mem_avail,
                "throttle": th,
            }
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            fh.flush()
            time.sleep(interval)
    print(f"Done: {out_path}")
    return 0


def _session_paths(metrics_dir: Path) -> List[Path]:
    return sorted(metrics_dir.glob("session_*.json"), key=lambda p: p.stat().st_mtime)


def cmd_bundle_recent(args: argparse.Namespace) -> int:
    fusion_root = resolve_fusion_root(Path(args.fusion_root) if args.fusion_root else None)
    metrics_dir = resolve_metrics_dir(fusion_root)
    if not metrics_dir.is_dir():
        print(f"ERROR: metrics dir not found: {metrics_dir}", file=sys.stderr)
        return 1
    sessions = _session_paths(metrics_dir)
    if not sessions:
        print(f"ERROR: no session_*.json under {metrics_dir}", file=sys.stderr)
        return 1
    take = sessions[-int(args.max_sessions) :]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    bundle_root = fusion_root / "field_runs" / "bundles" / f"snapshot_{stamp}"
    dest_metrics = bundle_root / "metrics"
    dest_metrics.mkdir(parents=True, exist_ok=True)

    copied: List[str] = []
    for sp in take:
        dst = dest_metrics / sp.name
        shutil.copy2(sp, dst)
        copied.append(sp.name)

    # Manifests: copy those newer than oldest bundled session (cheap heuristic)
    if take:
        t0 = take[0].stat().st_mtime
        md_dir = metrics_dir
        for mp in sorted(md_dir.glob("manifest_*.json")):
            if mp.stat().st_mtime >= t0 - 5.0:
                shutil.copy2(mp, dest_metrics / mp.name)

    # Run notes + monitors created during overlapping wall time
    for sub in ("run_notes", "monitors"):
        src = fusion_root / "field_runs" / sub
        if src.is_dir():
            dst = bundle_root / sub
            shutil.copytree(src, dst, dirs_exist_ok=True)

    readme = bundle_root / "README.txt"
    readme.write_text(
        "\n".join(
            [
                "SmartBinocular field bundle",
                f"created_utc: {stamp}",
                f"fusion_root: {fusion_root}",
                "sessions_copied:",
                *[f"  - {n}" for n in copied],
                "",
                "Pull to dev: use repo rsync.sh (pulls fusion_captures/ recursively).",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(f"Bundle written to {bundle_root}")
    return 0


def cmd_print_matrix(_args: argparse.Namespace) -> int:
    text = """
Thesis field matrix (run each row ≥5 min on RPi4; adjust ML/env as needed)

┌─────────┬──────────────────────────────────────────────────────────────┐
│ Row     │ Operator checklist                                           │
├─────────┼──────────────────────────────────────────────────────────────┤
│ fusion  │ ML_INFERENCE_ENABLED=1 env_mode=auto_rule python -m smartbinocular │
│         │ After boot: press 3 for fusion; run ≥300 s; quit with q.    │
├─────────┼──────────────────────────────────────────────────────────────┤
│ imx     │ Same command; press 1 (IMX / NIR path); ≥300 s.               │
├─────────┼──────────────────────────────────────────────────────────────┤
│ thermal │ Same command; press 2 (thermal path); ≥300 s.                 │
└─────────┴──────────────────────────────────────────────────────────────┘

Before each row (optional metadata file for prose / thesis captions):

  python tools/rpi_field_collect.py run-note --tag fusion_ml_on --notes "night indoor"

Parallel host telemetry (Terminal A):

  python tools/rpi_field_collect.py host-monitor --duration_sec 330 --interval_sec 5

After sessions:

  python tools/rpi_field_collect.py bundle-recent --max_sessions 12

Sync to dev: see rsync.sh (pull fusion_captures/, logs/ml/, data/eval/thermal_seq/).
"""
    print(text.strip())
    return 0


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    p_note = sub.add_parser("run-note", help="Write JSON run metadata before a manual session")
    p_note.add_argument("--tag", required=True, help="Short label, e.g. fusion_ml_on")
    p_note.add_argument("--notes", default="", help="Free-text operator notes")
    p_note.add_argument(
        "--fusion-root",
        default=None,
        help="Fusion captures root (default: ./fusion_captures or ~/fusion_captures)",
    )
    p_note.add_argument(
        "--capture-env",
        default="ML_INFERENCE_ENABLED,ML_MODEL_PATH,env_mode",
        help="Comma-separated env var names to snapshot into JSON",
    )
    p_note.set_defaults(func=cmd_run_note)

    p_mon = sub.add_parser("host-monitor", help="NDJSON CPU/mem/throttle sampling (no psutil required)")
    p_mon.add_argument("--interval_sec", type=float, default=5.0)
    p_mon.add_argument("--duration_sec", type=float, default=600.0)
    p_mon.add_argument("--fusion-root", default=None, help="Fusion captures root for output path")
    p_mon.set_defaults(func=cmd_host_monitor)

    p_b = sub.add_parser("bundle-recent", help="Copy recent session JSON (+ manifests, notes, monitors) for rsync")
    p_b.add_argument("--max_sessions", type=int, default=8)
    p_b.add_argument("--fusion-root", default=None)
    p_b.set_defaults(func=cmd_bundle_recent)

    p_m = sub.add_parser("print-matrix", help="Print thesis session matrix + reminders")
    p_m.set_defaults(func=cmd_print_matrix)

    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(list(argv) if argv is not None else None)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
