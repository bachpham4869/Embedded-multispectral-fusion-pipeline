# Deployment Hardening & Reproducibility Notes

*Archived 2026-04+ — still valid as reference; active runbook: [`../docs/RPi_RUNBOOK.md`](../../docs/RPi_RUNBOOK.md).*

Scope: software-side controls only. Hardware items (LUKS, Zymkey, mTLS, fail2ban, UFW, AIDE) are operational concerns outside the thesis scope.

---

## 1. Dependency Freeze (`uv sync --frozen`)

Always deploy with the exact dependency snapshot committed in `uv.lock`:

```bash
uv sync --frozen          # installs exact versions — no resolution
uv sync --frozen --extra ml-tools   # include sklearn/joblib for inference
```

Never use `uv sync` (without `--frozen`) on the RPi — version drift between training (Mac) and inference (RPi) can silently shift RF probabilities. The `sklearn_version` key in every model bundle documents the expected version; `EnvClassifier._load()` warns on mismatch.

---

## 2. Model Integrity (`model_registry.json`)

`models/train_classifier.py` writes `models/model_registry.json` after every training run. Each entry records:

```json
{
  "production/env_classifier.joblib": {
    "sha256": "<full-64-char-hex>",
    "promoted_from": "rf_phase1_retrain_optical12.joblib",
    "timestamp_iso": "2026-04-25T..."
  }
}
```

`EnvClassifier._load()` checks the SHA-256 on every startup (warn-only on mismatch — inference continues). To re-verify a deployed model manually:

```bash
python -c "
import hashlib, json
sha = hashlib.sha256(open('models/production/env_classifier.joblib','rb').read()).hexdigest()
reg = json.load(open('models/model_registry.json'))
key = 'env_classifier.joblib'
print('MATCH' if reg.get(key, {}).get('sha256') == sha else 'MISMATCH', sha[:16])
"
```

To deploy a new model via rsync after retraining on Mac:

```bash
rsync -av models/production/env_classifier.joblib models/model_registry.json pi@raspberrypi.local:~/smartBinocular/models/production/
```

---

## 3. JSONL Log Field Minimisation (GDPR Art. 5(1)(c))

The session JSONL logs (`logs/ml/*.jsonl`) must contain only the fields defined in `FeatureRecord.to_dict()`. Fields that must **not** appear:

- GPS coordinates or location identifiers
- Operator name (use `operator_id` from `config.py`'s `experiment.operator_id` — an opaque token)
- Raw images or facial features
- Any free-text notes that could identify a person

The `ML_LOG_INTERVAL` default (every 5 frames) combined with the 9 fps thermal rate means ~1.8 records/s — sufficient for ML training without capturing excessive personally-identifiable context.

---

## 4. Session Manifest (`manifest_<session_id>.json`)

Written at session start by `metrics_write_run_manifest()`. Contains:

| Field | Purpose |
|-------|---------|
| `session_id` | UUID + timestamp — links manifest to JSONL and capture images |
| `ml_model_sha256_short` | First 16 hex chars of model SHA-256 |
| `homography_quality` | `{max_corner_drift_px, all_corners_within_nir}` — first-order correctness check |
| `git_revision` | Short SHA of deployed commit |
| `stage_timing_ms` | Per-stage mean ± std for the session (P2-B) |
| `brisque_by_bucket` | BRISQUE mean ± std per night bucket (P2-A, only when `iqa_logging_enabled=True`) |
| `frames_by_env` | Frame count per ENV class — documents operating conditions |
| `frames_by_bucket` | Frame count per optical bucket |

---

## 5. BRISQUE Image Quality (P2-A)

BRISQUE requires two model files that do **not** ship in the `opencv-contrib-python` pip wheel. Download them from the OpenCV source tree and place in `models/brisque/`:

- `brisque_model_live.yml`
- `brisque_range_live.yml`

Source:
```
https://github.com/opencv/opencv_contrib/tree/master/modules/quality/samples
```

Then enable logging:
```python
# config.py
"iqa_logging_enabled": True,
```

Subsampled at 1/30 frames (~0.3 Hz at 9 fps thermal rate) — overhead is negligible. Report as a **comparative** quality indicator between night-relevant buckets A, B, D, E only. BRISQUE is calibrated on natural-colour images; do not use absolute scores for cross-dataset claims.

---

## 6. Per-Stage Latency (P2-B)

Stage timing is always active (zero-cost accumulator). Results appear in `session_summary.json` under `stage_timing_ms`:

```json
"stage_timing_ms": {
  "framecache":  {"mean_ms": 0.4,  "std_ms": 0.1, "n": 1800},
  "nir_bucket":  {"mean_ms": 6.2,  "std_ms": 1.3, "n": 1800},
  "thermal_proc":{"mean_ms": 2.1,  "std_ms": 0.4, "n": 200},
  "jerk":        {"mean_ms": 1.1,  "std_ms": 0.2, "n": 1800},
  "ml_infer":    {"mean_ms": 0.8,  "std_ms": 0.1, "n": 120},
  "blend":       {"mean_ms": 3.5,  "std_ms": 0.6, "n": 1800},
  "hud":         {"mean_ms": 0.9,  "std_ms": 0.1, "n": 1800},
  "display":     {"mean_ms": 1.2,  "std_ms": 0.3, "n": 1800}
}
```

Budget: sum of all stages ≤ 50 ms/frame. Run 300-frame profiling sessions for each `opt_profile` (`baseline`, `static_scan`, `handheld_pan`) on the deployed RPi4.

---

## 7. Epistemic Variance (P2-D)

After P1-A retraining with `CalibratedClassifierCV`, the RF inference thread computes the mean per-class variance across all trees and stores it in `MLSharedResult.get_epistemic_var()`. This is logged per inference cycle via the existing ML JSONL hook. High variance (>0.05) indicates the ensemble disagrees — useful for flagging low-confidence frames in the thesis evaluation.

Access path:
```python
evar = ml_shared_result.get_epistemic_var()  # Optional[float]
```

---

## 8. S4 Luminance-Cap Gate (`display_luma_cap_glare_gate`)

`RPI_THROUGHPUT_MAX_DEFAULTS` sets `display_luma_cap_glare_gate: True`. When enabled **and** `display_grade_mode='luma_only'` (also set in the same block), the inline gate in `main.py` skips the `display_luminance_cap_bgr()` BGR↔LAB round-trip on any frame where neither the NIR glare flag nor the thermal glare flag is raised. When `display_grade_mode='full'`, `display_grade_and_cap_bgr()` runs unconditionally and the flag has no effect. Field evidence (schema 1.4-opt, `session_20260424-144637.json`): `blend` stage dropped from ~40 ms → ~10–11 ms when the gate suppressed most non-glare frames.

**Trade-off:**

| Gate setting | `blend` cost | Luminance behaviour |
|---|---|---|
| `True` (default) | ~10–11 ms (non-glare frames) / ~38–42 ms (glare frames) | L-cap applied only when heuristic fires; subtle saturation events may pass through |
| `False` | ~38–42 ms every frame | L-cap always applied; consistent output but ~28–30 ms budget hit per frame |

**When to change:** Set `display_luma_cap_glare_gate: False` for controlled evaluation sessions where absolute luminance consistency is required (e.g. BRISQUE inter-bucket comparison). Leave `True` for all deployment and real-time operation profiles. The gate does **not** disable the L-cap permanently — it conditions it on the same glare signals already used to drive the HUD.

**Effect on thesis evaluation:** The `blend` timing in thesis Table 2 / P3-B latency figures should report the gated value (`~10 ms`) as the operational baseline; document the ungated cost as a footnote showing the cap's overhead when glare is active.

---

## Out-of-Scope Hardware Items

The following are operational hardening controls outside the thesis evaluation surface:

- Full-disk encryption (LUKS) on the SD card
- Hardware security module (Zymkey / TPM)
- Mutual TLS for any remote telemetry channel
- fail2ban / UFW firewall on SSH port
- File integrity monitoring (AIDE / Tripwire)
- ITAR / export control compliance
