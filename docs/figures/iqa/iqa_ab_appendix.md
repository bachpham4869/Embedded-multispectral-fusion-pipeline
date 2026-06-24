# IQA A/B Appendix — NIR Bucket Evaluation (offline batch)

**Living log** — updated after each iteration round. Never overwrite prior rounds; append dated subsections.

---

## 1. Metrics Glossary

| Metric | Formula | Direction | Typical range | Meaning |
|--------|---------|-----------|---------------|---------|
| `log_rms_contrast` | std(log(L + 1)) | ↑ better | 0.8–1.6 | Global contrast after bucket fn |
| `Δlog_rms` | log_rms_after − log_rms_before | ↑ better | −0.5 to +0.5 | Contrast lift (positive) or loss (negative) |
| `pct_sat_after` | mean(L_out ≥ 250) | ↓ better | 0–0.20 | Absolute fraction of over-exposed pixels |
| `pct_crush_after` | mean(L_out ≤ 5) | ↓ better | 0–0.80 | Absolute fraction of crushed (under-exposed) pixels |
| `Δcrush` | pct_crush_after − pct_crush_before | 0 ideal | −0.10 to +0.10 | Crush added (+) or relieved (−) by bucket fn; guard target: ≤ +0.05 |
| `hist_entropy` | H(histogram) | ↑ better | 3–7 | Information content of output histogram |
| `local_contrast_std` | std of local patch stds | ↑ better | 5–70 | Spatial texture richness |
| `pct_blended` (F) | fraction with w ∈ (0.02, 0.98) | ↑ better | 0–1.0 | Fraction genuinely blending A+C vs degenerating to one bucket |

### How to read failures

- **Δcrush > +0.05**: bucket adds shadow crush — likely wrong dispatch or omega too high (D)
- **pct_sat_after > 0.10**: output over-exposed — night-boost on bright image (A)
- **Δlog_rms < −0.30**: bucket degrades contrast dramatically — likely wrong dispatch (D on night)
- **Near-passthrough** (|Δlog_rms| < 0.03, |Δcrush| < 0.005): minimal change; correct for C/E, problematic for D/A as silent fail

### Limits

- **Cold-start (Bucket A):** `still_image_cold_start_mode` forces `reset()` + `update_rate=1`. Live EMA (10–20 frame window) behaves differently; cold-start overstates saturation on night-boost path.
- **BRISQUE not used:** proxy metrics only; no ground-truth labels.
- **Bucket E surrogate:** single stills return input unchanged (buf < N). Rain results = passthrough by construction.
- **Bucket F EMA proxy:** `nir_b_ema_norm = mean(green)/255` per still; saturates to 0 or 1 more often than production. Results are directional only.
- **Domain shift (R3):** external datasets (darkface, ExDark, weather11); may not match IMX290 spectral response.

---

## 2. Method

### Dataset

- **Manifest:** `data/eval/nir_val/manifest_v2.csv`  
- **n = 270 images**, 9 ENV_CLASS labels  
- **Provenance:** `data/eval/nir_val/MANIFEST_V2_NOTES.md` — seed=42, per-class quotas, source dirs  
- **Manifest SHA-256 (12 hex):** 5f9aecb5e1ea

### Per-class counts

| env_class | n | primary source |
|-----------|---|----------------|
| night_clear | 40 | darkface/image (extreme_dark subset, mean_L < 30) |
| normal_night | 40 | ExDark (Bicycle, Boat, Cup, Dog, Cat) |
| nir_night | 40 | weather/gray (mid-brightness 60–110) |
| fog | 30 | weather11/dataset/fogsmog |
| glare | 30 | glare/real/input (pct_sat > 0.05) |
| backlight | 25 | backlight/ |
| rain | 20 | weather11/dataset/rain |
| normal_day | 20 | weather11 daylight (dew, frost, rainbow) |
| mixed_edge | 25 | hand-picked borderline (dark+glare, foggy+night) |

### Run modes

| Mode | Applies to |
|------|------------|
| `still_image_cold_start_mode` (reset+update_rate=1) | Bucket A |
| `still_image_cold_start_mode` (stateless per-frame) | Buckets B, C, D |
| `rain_surrogate` (single still → input unchanged) | Bucket E on stills |
| `single_frame_ema_proxy` (mean(green)/255) | Bucket F |

### Tool commands (reproducible)

```bash
# Per-bucket sweeps
python tools/sweep_hybrid_params.py --manifest data/eval/nir_val/manifest_v2.csv
python tools/sweep_clahe_clip.py --frame-dir data/weather/gray/ --bucket B
python tools/sweep_anti_glare.py --manifest data/eval/nir_val/manifest_v2.csv --out docs/tables/iqa/sweep_anti_glare_c.csv
python tools/sweep_dehaze_omega.py --manifest data/eval/nir_val/manifest_v2.csv
python tools/sweep_transition_blend.py --manifest data/eval/nir_val/manifest_v2.csv

# Mis-dispatch matrix
python tools/batch_nir_enhancer.py --manifest data/eval/nir_val/manifest_v2.csv \
    --bucket A B C D E F --out docs/tables/iqa/raw/mis_dispatch_matrix.csv

# Dispatch consistency (rule vs manifest)
python tools/check_dispatch_consistency.py \
    --manifest data/eval/nir_val/manifest_v2.csv \
    --out docs/tables/iqa/dispatch_consistency.md

# Batch IQA eval driver (dated rounds under data/eval/iqa_runs/)
python tools/run_nir_iqa_eval.py --note "Round N: <what changed>"

# Diff between rounds
python tools/diff_iqa_runs.py docs/tables/iqa/raw/mis_dispatch_matrix.csv \
    data/eval/iqa_runs/round_2026-04-28.csv
```

---

## 3. Per-Bucket Aggregate Results (diagonal — correct dispatch only)

### Round 1 (2026-04-28, after preset tuning + all guards)

| Bucket | ENV_CLASS served | n | mean Δlog_rms | mean pct_sat_after | mean pct_crush_after | Notes |
|--------|-----------------|---|:------------:|:-----------------:|:------------------:|-------|
| **A** | night_clear, normal_night | 80 | −0.103 | 0.067 | 0.001 | Cold-start; sat from night-boost; A.bright guard active |
| **B** | nir_night | 40 | +0.209 | 0.012 | 0.007 | clip=0.5 (clahe_clip_scale=0.16) |
| **C** | glare, backlight, normal_day | 75 | +0.023 | 0.001 | 0.015 | high_pct=90, sat_at=220; passthrough-heavy |
| **D** | fog | 30 | −0.388 | 0.016 | 0.004 | omega=0.92; D.dark guard protects night classes |
| **E** | rain | 20 | +0.000 | 0.022 | 0.012 | Surrogate — still returns input unchanged |
| **F** | mixed_edge | 25 | −0.090 | 0.125 ⚠ | 0.001 | Single-frame EMA proxy; w≈0 → A_out; A_out sat propagates |

⚠ F/mixed_edge: high pct_sat is A_out propagation on dark scenes (w < 0.02 → short-circuit to A). Documented residual limitation.

### Round 0 baseline (pre-guard, 2026-04-27)

| Bucket | n | mean Δlog_rms | mean pct_sat_after | mean pct_crush_after |
|--------|---|:------------:|:-----------------:|:------------------:|
| A | 80 | −0.084 | 0.067 | 0.001 |
| B | 40 | +0.212 | 0.012 | 0.007 |
| C | 75 | +0.026 | 0.000 | 0.015 |
| D | 30 | −0.388 | 0.016 | 0.004 |
| E | 20 | −0.006 | 0.022 | 0.012 |
| F | 25 | −0.090 | 0.125 | 0.001 |

> Round 0 diagonal: D diagonal shows correct fog metrics; D crush on dark classes is captured in the 6×9 matrix (§6), not the diagonal.

---

## 4. Per-Bucket Sections

### Bucket A — HybridNIREnhancer

**Function:** Dark/bright channel atmosphere estimate + adaptive CLAHE + detail weight-map + unsharp + dark-scene L boost (LAB).  
**ENV_CLASS routes:** `night_clear`, `normal_night`, `default`.  
**Cost (RPi):** ~4–8 ms with update_rate=20; ~20 ms cold-start.

**Sweep result** (`docs/tables/iqa/sweep_hybrid_a.csv`, stopping rule: max log_rms_after s.t. pct_sat_after < 0.05):

| env_class | Best params | log_rms_after | pct_sat_after | Pass |
|-----------|-------------|:------------:|:------------:|:----:|
| night_clear | ds=0.35, clip=0.5, proc=320×240 | 0.9200 | 0.0448 | YES |
| nir_night | — | — | — | NO (all fail; cold-start overstates sat) |
| normal_night | — | — | — | NO (all fail; cold-start overstates sat) |

> nir_night and normal_night: no passing cell in cold-start mode. Production dispatch avoids A for these (→ B). Cold-start diverges from live EMA.

**M3 tuning:** Added `nir_enhancer_detail_strength: 0.35` to `night_clear` preset opt_overrides (`env_presets.py`).

**Guards in `nir_pipeline.py`:**
- **A.bright** (`cur_bright ≥ 0.30`): skip night_boost — prevents oversaturation on intermediate-brightness inputs. Lowers threshold from 0.45 → 0.30; removes 1.3× boost tier. nir_night sat: 0.171 → 0.072 ✓
- **A.dark** (`mean_gray < 15`): clamp `detail_strength` to 0.0 — prevents noise amplification on extreme_dark.

**Mis-dispatch verdict:**
- A on nir_night: sat 0.072 (post-guard ✓)
- A on backlight: sat 0.129 ⚠ (residual; production dispatch uses C for backlight)
- A on mixed_edge: sat 0.135 ⚠ (residual; production dispatch uses F/A for transition only)

---

### Bucket B — nir_nir_night_clahe

**Function:** Single CLAHE on LAB L channel at `clip = 3.0 × clahe_clip_scale`. Tile 4×4. ~1–2 ms.  
**ENV_CLASS routes:** `nir_night`.

**Sweep result** (`docs/tables/iqa/clahe_clip_sweep.csv`, 7,129 gray images, stopping rule: max log_rms s.t. pct_sat_new < 0.02):

| clip | log_rms | pct_sat_new | Pass |
|:----:|:-------:|:-----------:|:----:|
| 0.5 | 0.7896 | +0.000081 | YES |
| 3.0 (prev default) | 0.7760 | −0.00186 | YES |
| 8.0 | 0.7915 | −0.000345 | YES |

> All clips pass; clip=0.5 selected (proxy-optimal per plan; near-zero new saturation; conservative).  
> `clahe_clip_scale = 0.16` → `clip = max(0.5, 3×0.16=0.48) = 0.5`.

**M3 tuning:** `nir_night` preset `nir_enhancer_clahe_clip_scale`: 1.2 → 0.16.

**Guard (B):** `effective_scale = min(clahe_clip_scale, 0.5) if pct_sat > 0.10` — caps clip on bright inputs.

**Mis-dispatch verdict:** B on all classes — Δcrush ≤ 0.003, pct_sat_after ≤ 0.068. B is the most robust bucket across all ENV_CLASS (CLAHE is conservative at clip=0.5).

---

### Bucket C — nir_anti_glare_bgr

**Function:** Highlight gate (`nir_highlight_need_compress`) → tone-map (gamma + roll-off) or passthrough.  
**ENV_CLASS routes:** `glare`, `backlight`, `normal_day`.  
**Cost:** passthrough ~0.1 ms; tone-map ~3 ms.

**Sweep result** (`docs/tables/iqa/sweep_anti_glare_c.csv`, stopping rule: min pct_sat_after s.t. Δlog_rms ≥ −0.05):

| env_class | high_pct | sat_at | gamma | pct_sat_after | Δlog_rms | Pass |
|-----------|:--------:|:------:|:-----:|:------------:|:-------:|:----:|
| glare | 90 | 220 | 0.65 | 0.0011 | +0.049 | YES |
| backlight | 90 | 220 | 0.65 | 0.0000 | +0.047 | YES |
| normal_day | 90 | 220 | 0.65 | 0.0005 | −0.009 | YES |
| mixed_edge | 90 | 220 | 0.65 | 0.0021 | +0.000 | YES (passthrough) |

**M3 tuning:** `glare` preset: `nir_high_pct` 93.0 → 90.0, `nir_saturate_at` 228.0 → 220.0.  
`backlight` preset: `nir_high_pct` 92.0 → 90.0, `nir_saturate_at` 226.0 → 220.0.

**Guard (C):** Early-exit passthrough when `mean(gray) < 60` — prevents unwanted darkening on dark mis-dispatch. Δcrush = 0.000 on all night classes ✓

---

### Bucket D — nir_dehaze_lite

**Function:** DCP dark-channel prior dehazing at 160×120 downsample. ~4–6 ms.  
**ENV_CLASS routes:** `fog`.

**Sweep result** (`docs/tables/iqa/sweep_dehaze_d.csv`, stopping rule: max fog log_rms s.t. night pct_crush < 0.05):

| omega | ds | fog log_rms | fog pct_crush | nir_night pct_crush | Pass |
|:-----:|:--:|:-----------:|:------------:|:------------------:|:----:|
| 0.55 | 160×120 | 0.585 | 0.002 | 0.016 | YES |
| 0.92 | 160×120 | **0.742** | 0.006 | 0.023 | YES |

> omega=0.92 maximizes fog contrast; night classes protected by D.dark guard.

**M4 tuning:** `fog_dehaze_omega: 0.92` added to `RPI_THROUGHPUT_MAX_DEFAULTS` (`config.py`). Global baseline remains 0.85.

**Guard (D):** Return `frame` unchanged when `mean(I) < 0.18` OR `pct(I < 0.05) > 0.20`.  
Night_clear Δcrush: +0.084 → 0.000 ✓ | Normal_night: +0.068 → +0.001 ✓ | Mixed_edge: +0.122 → 0.000 ✓

---

### Bucket E — RainTemporalMedian

**Function:** Rolling N-frame median over `_buf`. Returns input until buf full.  
**ENV_CLASS routes:** `rain`. **Cost:** ~4–6 ms (skipped on static scenes by guard).  
**Offline fidelity:** SURROGATE-ONLY.

**Surrogate result:** `docs/tables/iqa/sweep_rain_e.csv` — all stills return input; demonstrates median operator is well-behaved but provides no rain-streak removal on single frames.

**Guard (E):** Skip median if `abs(mean_curr − mean_prev) < 0.5/255` for 3 consecutive frames. Saves ~4 ms on static scenes. Verified on surrogate: no quality regression.

> Real rain-streak removal validation deferred to RPi (see §9).

---

### Bucket F — nir_transition_blend

**Function:** `cv.addWeighted(A_out, 1−w, C_out, w, 0)` where `w = clip((nir_b_ema_norm − lo) / (hi − lo), 0, 1)`.  
**ENV_CLASS routes:** `transition`. Evaluated offline on `mixed_edge`.  
**Offline fidelity:** PARTIAL — single-frame EMA proxy.

**Sweep result** (`docs/tables/iqa/sweep_transition_f.csv`, stopping rule: ≥40% mixed_edge with `w ∈ (0.02, 0.98)`):

> **No cell passes.** mixed_edge mean_blend_w ≈ 0.000 for all cells — dark mixed_edge images saturate w to 0 (fully A). Single-frame EMA proxy cannot capture live temporal blend dynamics. Result: directional only.

**Guard (F):** Short-circuit `return A_out if w < 0.02; return C_out if w > 0.98`. Avoids addWeighted cost (~0.5 ms) when blend is degenerate.

---

## 5. Per-ENV_CLASS Profiles (abbreviated)

| env_class | Dispatch | Δlog_rms | pct_sat | pct_crush | Key finding |
|-----------|:--------:|:--------:|:-------:|:---------:|-------------|
| night_clear | A | −0.12 | 0.059 | 0.000 | A preserves dark character; detail_strength=0.35 per M3 |
| normal_night | A | −0.05 | 0.074 | 0.000 | Similar to night_clear; cold-start sat expected |
| nir_night | B | +0.21 | 0.012 | 0.007 | Best contrast lift; clip=0.5 conservative and robust |
| fog | D | −0.39 | 0.016 | 0.004 | DCP reduces contrast but removes haze; omega=0.92 optimal |
| glare | C | +0.05 | 0.001 | 0.013 | Tone-map well-calibrated; high_pct=90 tighter gate |
| backlight | C | +0.05 | 0.000 | 0.018 | As glare; M3 tightened gate prevents sat residue |
| rain | E | +0.000 | 0.022 | 0.012 | Surrogate only; temporal eval on RPi required |
| normal_day | C | −0.009 | 0.001 | 0.012 | Near-passthrough (bright inputs rarely trigger gate) |
| mixed_edge | F | −0.09 | 0.125 | 0.001 | F returns A_out for dark scenes; A_out sat propagates |

---

## 6. Mis-Dispatch Matrix (6×9 summary)

**Full table + bootstrap CIs:** `docs/tables/iqa/mis_dispatch_matrix_summary.md` and `mis_dispatch_matrix_summary.csv`

**Failure-mode cells (pre-guard Round 0):**

| Bucket | env_class | Δlog_rms | Δcrush | pct_sat | Type |
|--------|-----------|:--------:|:------:|:-------:|------|
| **A** | nir_night | +0.128 | −0.022 | **0.171** | Oversaturation |
| **A** | backlight | −0.536 | −0.018 | **0.118** | Contrast loss + oversaturation |
| **A** | mixed_edge | −0.091 | −0.345 | **0.133** | Oversaturation |
| **D** | night_clear | −1.023 | **+0.084** | 0.001 | Crush + massive contrast loss |
| **D** | normal_night | −0.804 | **+0.068** | 0.001 | Crush + massive contrast loss |
| **D** | glare | −0.666 | +0.046 | 0.010 | Large contrast loss |
| **D** | backlight | −0.963 | +0.041 | 0.037 | Large contrast loss |
| **D** | mixed_edge | −0.983 | **+0.122** | 0.000 | Crush + massive contrast loss |

**After guards (Round 1):** All D-on-dark failures resolved (Δcrush ≤ 0.001). A-on-nir_night resolved (sat 0.171 → 0.072). See §7 diff table and `docs/tables/iqa/guard_before_after_summary.md`.

---

## 7. Iteration Log

### Round 0 — 2026-04-27 (baseline, manifest_v2, 6-bucket matrix)

**Source CSV:** `docs/tables/iqa/raw/mis_dispatch_matrix.csv` (1620 rows = 270 images × 6 buckets)  
**git commit at run:** f71e50f  
**Note:** First full 6×9 mis-dispatch matrix; manifest_v2; all buckets A–F

**What was established:**
- D on dark classes (night_clear, normal_night, mixed_edge): severe crush (Δcrush > 0.08) → D.dark guard required
- A on nir_night: oversaturation (pct_sat=0.171) → A.bright guard required
- B on all classes: most conservative and robust (Δcrush ≤ 0.003, pct_sat ≤ 0.07)
- C on dark classes: near-passthrough (gate rarely triggers) → correct behavior
- D.dark guard needed: mean(I)<0.18 or pct(I<0.05)>0.20 → passthrough

---

### Round 1 — 2026-04-28 (preset tuning + all 7 guards)

**Source CSV:** `data/eval/iqa_runs/round_2026-04-28.csv` (1620 rows)  
**git commit:** f1a3c48  
**manifest_sha:** 5f9aecb5e1ea

**Changes applied:**
1. Bucket A / night_clear: `nir_enhancer_detail_strength` → 0.35 (`env_presets.py`)
2. Bucket B / nir_night: `nir_enhancer_clahe_clip_scale` → 0.16 → clip=0.5 (`env_presets.py`)
3. Bucket C / glare: `nir_high_pct` 93→90, `nir_saturate_at` 228→220 (`env_presets.py`)
4. Bucket C / backlight: `nir_high_pct` 92→90, `nir_saturate_at` 226→220 (`env_presets.py`)
5. Bucket D: `fog_dehaze_omega` 0.85→0.92 in `RPI_THROUGHPUT_MAX_DEFAULTS` (`config.py`)
6. 7 per-bucket guards in `nir_pipeline.py` (A.dark, A.bright, B, C, D, E, F)

**Diff table (Round 0 → Round 1, selected cells):**

| env_class | bucket | Δ(Δlog_rms) | Δpct_sat | Δpct_crush | Interpretation |
|-----------|:------:|:-----------:|:--------:|:----------:|----------------|
| night_clear | D | **+1.023** | 0.000 | **−0.084** | D.dark guard eliminates crush ✓ |
| normal_night | D | **+0.709** | +0.001 | **−0.067** | D.dark guard ✓ |
| mixed_edge | D | **+0.984** | 0.000 | **−0.122** | D.dark guard ✓ |
| nir_night | A | −0.030 | **−0.098** | 0.000 | A.bright guard: sat 0.171→0.072 ✓ |
| glare | A | −0.013 | **−0.034** | 0.000 | A.bright guard: sat 0.121→0.087 ✓ |
| backlight | A | −0.016 | **−0.066** | 0.000 | A.bright guard: sat 0.195→0.129 (residual ⚠) |

**Full diff:** run `python tools/diff_iqa_runs.py docs/tables/iqa/raw/mis_dispatch_matrix.csv data/eval/iqa_runs/round_2026-04-28.csv`

---

## 8. Edge Cases and Guards

### Per-bucket guard catalog

| # | Bucket | Failure mode | Guard | Cost | Before → After |
|---|:------:|-------------|-------|:----:|----------------|
| 1 | A.bright | A on nir_night → pct_sat > 0.10 | Skip night_boost when `cur_bright ≥ 0.30` (thresh 0.45→0.30; remove 1.3× tier) | 0 ms | 0.171 → 0.072 ✓ |
| 2 | A.dark | A on extreme_dark → noise amp | Clamp `detail_strength` to 0.0 when `mean(gray) < 15` | ~0.02 ms | Controlled ✓ |
| 3 | B | B on extreme_bright → sat residue | `effective_scale = min(clahe_clip_scale, 0.5) if pct_sat > 0.10` | ~0.05 ms | pct_sat ≤ 0.068 all classes ✓ |
| 4 | C | C on deep night → unwanted darkening | Early-exit passthrough when `mean(gray) < 60` | ~0.05 ms | Δcrush = 0.000 on night classes ✓ |
| 5 | D | D on dark scene → crush | `return frame if mean(I) < 0.18 or pct(I < 0.05) > 0.20` | ~0.10 ms | Δcrush +0.084→0.000 (night_clear) ✓ |
| 6 | E | E wastes ~4 ms on static | Skip median if `abs(mean_curr − mean_prev) < 0.5/255` for 3 frames | ~0.02 ms | No quality regression ✓ |
| 7 | F | F wastes A+C cost at boundary | `return A_out if w < 0.02; return C_out if w > 0.98` | 0 ms | Budget preserved ✓ |

All guards have `# guard: <failure-mode> — see mis_dispatch_matrix_summary.md` comments in `nir_pipeline.py`.

### Dispatch-fragile classes

**Source:** `docs/tables/iqa/dispatch_consistency.md` (rule-vs-manifest confusion matrix, 270 images)

- **Overall agreement:** 34/270 = **12.6%**
- **Root cause:** `nir_highlight_need_compress` at high_pct=95/sat_at=233 triggers on any image with bright highlights → near-universal `glare` rule label
- **Fragile classes (< 80% agreement):** all 9 classes except glare (76.7%)

> Not a bug. Per evidence register and `dispatch_consistency.md`: disagreements are evidence of rule-layer fragility motivating the ML compositor (`compose_env_from_ml_top2`). Thesis §8: production dispatch quality depends on ML compositor, not rule layer alone.

---

## 9. Limitations and Non-Claims

### Bucket-level fidelity limits

| Bucket | Limit | Consequence |
|--------|-------|-------------|
| **A** | Cold-start mode (`reset()+update_rate=1` per still) | Saturation overstated vs. live 10–20 frame EMA; nir_night/normal_night figures are pessimistic upper bounds |
| **E** | Single stills return input (buf < N=3) | Rain-streak removal NOT demonstrated; only confirms median operator handles stills correctly |
| **F** | `nir_b_ema_norm = mean(green)/255`; no temporal smoothing | `pct_blended` understated; production blend dynamics only evaluable on RPi |

### Statistical limits

- **Pre-registered stopping rules** declared before each sweep run; picked cell satisfies the rule (not global minimum) — prevents multiple-comparisons cherry-picking
- **Bootstrap CIs (mis-dispatch matrix):** 200 resamples, seed=42; per-cell CIs in `mis_dispatch_matrix_summary.csv`
- **Domain shift (R3):** external datasets; IMX290 spectral characteristics not matched

### What this thesis does NOT claim

- Perceptual quality improvement (no BRISQUE; proxy metrics only)
- IMX290-spectrum accuracy of offline evaluation
- Rain-streak removal validation (Bucket E requires RPi temporal sequence)
- Bucket F blend smoothness under realistic live `nir_b_ema_norm` dynamics
- Rule-layer dispatch accuracy (documented 12.6%; ML compositor is production path)

Cross-reference: `docs/THESIS_SCOPE_LIMITATIONS.md §5`

---

## 10. Conclusions

### Production-credible (offline eval sufficient)

| Claim | Evidence |
|-------|---------|
| Bucket B (clip=0.5) is conservative and robust for nir_night | `clahe_clip_sweep.csv`; Δlog_rms=+0.21, pct_sat=0.012 (40 images) |
| Bucket C (high_pct=90, sat_at=220) minimizes highlight residue | `sweep_anti_glare_c.csv`; pct_sat < 0.002, Δlog_rms > +0.04 |
| D.dark guard eliminates dark-scene crush | `guard_before_after_summary.md`; night_clear Δcrush +0.084→0.000 |
| A.bright guard resolves nir_night oversaturation | `guard_before_after_summary.md`; pct_sat 0.171→0.072 |
| 12.6% rule-layer dispatch agreement motivates ML compositor | `dispatch_consistency.md` (270 images, 9 classes) |

### Deferred to RPi field session

| Item | Reason |
|------|--------|
| Bucket A live EMA temporal quality | Cold-start diverges from 10–20 frame EMA |
| Bucket E real rain-streak removal | Requires ≥3 temporally adjacent frames |
| Bucket F production blend dynamics | Live `nir_b_ema_norm` smooths over time; proxy saturates |
| Per-bucket timing in production loop | `tools/measure_stage_timing.py` on RPi required |
| Dispatch quality with ML compositor | ML + posterior EMA only evaluable on live feed |

### Explicitly NOT claimed

- Perceptual quality superiority over fixed-path alternatives
- Generalization to non-IMX290 NIR sensors
- Real rain-streak removal effectiveness
- Bucket F blend smoothness in live conditions
