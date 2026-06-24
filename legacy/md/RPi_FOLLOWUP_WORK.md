# Raspberry Pi — follow-up work (post dev-machine audit)

*Archived 2026-04+ — chiến dịch test thiết bị đã chạy xong; checklist tóm tắt giữ lại tham chiếu.*

Các lệnh **rsync / SSH / chạy / pull metrics** tóm tắt ở **[`../../docs/RPi_RUNBOOK.md`](../../docs/RPi_RUNBOOK.md)**.

Work that **must run on the physical RPi** (or a device-identical image) with cameras, SPI thermal, and production-like config. The dev machine cannot replace this list.

---

## 1. Timing & Phase 3b (from `PIPELINE_EVIDENCE_REGISTER.md` A.3)

| Step | Action | Artifact / success |
| --- | --- | --- |
| 1 | Run the fusion pipeline in a representative scene (night + fusion mode) for **≥5 minutes** with metrics/session JSON enabled. | `fusion_captures/metrics/session_*.json` with `stage_timing_ms` / `fuse_stage_timing_ms`. |
| 2 | Copy session JSON to the dev machine (or path visible to repo) and run `python tools/measure_stage_timing.py` | Updates `docs/tables/timing/stage_timing_summary.{csv,md}` (paths relative to repo root) |
| 3 | If optimizing: after each change (fusion sub-profiling, resolution, flags from `THESIS_IMPROVEMENT_PLAN` Phase 3b), **repeat 1–2** to show before/after ms. | Thesis can cite measured p50/p95, not only design targets. |
| 4 | Align `README.md` claims with data: if `fusion_composite` p50 still exceeds the nominal budget, state **target vs measured** explicitly. | Honest limit section for defense. |

---

## 2. ENV ML inference (observe / promote)

| Step | Action |
| --- | --- |
| A | `ML_INFERENCE_ENABLED=1` with `ML_MODEL_PATH=models/production/env_classifier.joblib` (see [`DEPLOY_ML_MODEL.md`](DEPLOY_ML_MODEL.md) in this folder). |
| B | T032-style checklist: startup log, `debug=True` HUD, JSONL with `ml_env_label` / `ml_confidence`, no unhandled exceptions. |
| C | If promoting beyond observe-only, follow product gates in [`THESIS_IMPROVEMENT_PLAN.md`](THESIS_IMPROVEMENT_PLAN.md) / [`DEPLOY_HARDENING.md`](DEPLOY_HARDENING.md) (registry SHA, `uv sync --frozen` on RPi). |

---

## 3. Optional sweeps needing device or staged data

| Item | When |
| --- | --- |
| `sweep_clahe_clip.py` with real NIR crops | If tightening CLAHE clip bounds beyond PARTIAL / engineering notes in the register. |
| `sweep_kalman_qr.py` | After a **staged** thermal frame sequence is available under `data/` or `logs/` (see script docstring). |
| Re-run `threshold_sweep.py` / `secondary_threshold_sweep.py` | After retrain; then update `docs/tables/ml/*.csv`, `models/production/env_classifier.json` `ml_gate_reference`, and consistency tests. |

---

## 4. Housekeeping on Pi

- `rsync` updated `env_classifier.joblib` + `env_classifier.json` when training promotes a new bundle.
- Recompute registry SHA on Mac and verify `tests/test_model_registry_integrity.py` on a machine that has the joblib file.

---

*Complements: [`../../docs/PIPELINE_EVIDENCE_REGISTER.md`](../../docs/PIPELINE_EVIDENCE_REGISTER.md) (Parts A–D), [`THESIS_IMPROVEMENT_PLAN.md`](THESIS_IMPROVEMENT_PLAN.md) (Phase 3b, archive), [`DEPLOY_ML_MODEL.md`](DEPLOY_ML_MODEL.md).*
