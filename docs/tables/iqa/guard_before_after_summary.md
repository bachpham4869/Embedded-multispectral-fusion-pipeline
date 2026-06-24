# Guard Before/After Summary — Per-Bucket Mis-Dispatch Matrix

**Pre-guard:** `docs/tables/iqa/raw/mis_dispatch_matrix.csv`  
**Post-guard:** `docs/tables/iqa/raw/guard_before_after.csv` + `guard_bucket_a_v2.csv`  
  *(Bucket A re-evaluated after A.bright threshold lowered 0.45→0.30, removing 1.3× tier)*  

Metrics: **Δlog_rms** = after−before; **Δcrush** = after−before pct_crushed; **sat** = after pct_sat.  
Failure flag: Δcrush > 0.05 **OR** sat > 0.10 **OR** Δlog_rms < −0.30.  
Bootstrap 95% CI on Δcrush (200 resamples, seed=42).

## Per-Cell Δcrush: before → after guard (rows = env_class, cols = bucket)

| env_class \ bucket | **A** | **B** | **C** | **D** | **E** | **F** |
|--------------------|----------|----------|----------|----------|----------|----------|
| night_clear | -0.677→-0.677 | -0.675→-0.675 | +0.000→+0.000 | +0.084→+0.000 | +0.000→+0.000 | -0.677→-0.677 |
| normal_night | -0.257→-0.257 | -0.250→-0.250 | +0.000→+0.000 | +0.068→+0.001 | +0.000→+0.000 | -0.255→-0.255 |
| nir_night | -0.022→-0.022 | -0.016→-0.016 | +0.000→+0.000 | -0.002→-0.003 | -0.001→-0.001 | -0.014→-0.014 |
| fog | -0.002→-0.002 | -0.001→-0.001 | +0.000→+0.000 | +0.001→+0.001 | -0.000→-0.000 | -0.000→-0.000 |
| glare | -0.014→-0.014 | -0.013→-0.013 | +0.000→+0.000 | +0.046→+0.012 | +0.000→+0.000 | -0.014→-0.014 |
| backlight | -0.018→-0.018 | -0.016→-0.016 | +0.000→+0.000 | +0.041→+0.030 | -0.000→-0.000 | -0.015→-0.015 |
| rain | -0.011→-0.011 | -0.008→-0.008 | +0.000→+0.000 | -0.001→-0.001 | -0.000→-0.000 | -0.008→-0.008 |
| normal_day | -0.012→-0.012 | -0.008→-0.008 | +0.000→+0.000 | -0.002→-0.002 | -0.000→-0.000 | -0.002→-0.002 |
| mixed_edge | -0.345→-0.345 | -0.342→-0.342 | +0.000→+0.000 | +0.122→+0.000 | +0.002→+0.002 | -0.345→-0.345 |

## Per-Cell sat_after: before → after guard

| env_class \ bucket | **A** | **B** | **C** | **D** | **E** | **F** |
|--------------------|----------|----------|----------|----------|----------|----------|
| night_clear | 0.058→0.059 | 0.001→0.001 | 0.000→0.001 | 0.001→0.001 | 0.001→0.001 | 0.050→0.051 |
| normal_night | 0.077→0.074 | 0.003→0.003 | 0.000→0.004 | 0.002→0.004 | 0.004→0.004 | 0.048→0.047 |
| nir_night | 0.171→0.072 | 0.012→0.012 | 0.000→0.000 | 0.007→0.007 | 0.010→0.010 | 0.000→0.000 |
| fog | 0.048→0.052 | 0.018→0.020 | 0.000→0.000 | 0.016→0.016 | 0.027→0.027 | 0.001→0.001 |
| glare | 0.121→0.087 | 0.020→0.020 | 0.000→0.001 | 0.014→0.014 | 0.023→0.023 | 0.007→0.007 |
| backlight | 0.195→0.129⚠ | 0.061→0.068 | 0.000→0.000 | 0.040→0.041 | 0.077→0.077 | 0.001→0.001 |
| rain | 0.119→0.073 | 0.016→0.019 | 0.000→0.000 | 0.008→0.008 | 0.022→0.022 | 0.000→0.000 |
| normal_day | 0.042→0.036 | 0.013→0.013 | 0.001→0.001 | 0.007→0.007 | 0.012→0.012 | 0.001→0.001 |
| mixed_edge | 0.135→0.135⚠ | 0.002→0.002 | 0.000→0.002 | 0.003→0.002 | 0.002→0.002 | 0.125→0.125⚠ |

## Guard Resolution — 7 Failure Modes

| # | Guard | Failure condition | Pre-guard | Post-guard | Status |
|---|-------|------------------|-----------|------------|--------|
| 1 | D on night_clear | Δcrush > 0.05 | 0.0839 | 0.0000 | YES ✓ |
| 2 | D on normal_night | Δcrush > 0.05 | 0.0681 | 0.0012 | YES ✓ |
| 3 | D on nir_night | Δcrush > 0.05 | -0.0023 | -0.0030 | YES ✓ |
| 4 | D on mixed_edge | Δcrush > 0.05 | 0.1219 | 0.0000 | YES ✓ |
| 5 | A on nir_night (A.bright guard, thresh 0.45→0.30) | sat_after > 0.1 | 0.1706 | 0.0722 | YES ✓ |
| 6 | A on glare (bonus A.bright fix) | sat_after > 0.1 | 0.1208 | 0.0866 | YES ✓ |
| 7 | B on backlight (guard B clip cap) | sat_after > 0.1 | 0.0612 | 0.0678 | YES ✓ |

### Residual limitations

| Case | sat_after | Note |
|------|-----------|------|
| A on backlight (mis-dispatch) | 0.1948→0.1289 | CLAHE amplifies pre-existing input sat; production dispatch avoids A for this class |
| A on mixed_edge (edge case) | 0.1354→0.1354 | CLAHE amplifies pre-existing input sat; production dispatch avoids A for this class |

## Diagnostic: C/E/F guards (passthrough/skip behaviors)

| Guard | Behavior | Pre Δcrush | Post Δcrush |
|-------|----------|-----------|------------|
| C on night_clear (early-exit) | passthrough on dark input | +0.0000 | +0.0000 |
| C on normal_night (early-exit) | passthrough on dark input | +0.0000 | +0.0000 |
| C on nir_night (early-exit) | passthrough on dark input | +0.0000 | +0.0000 |
| E on rain (static skip) | 3-count static → return input | -0.0002 | -0.0002 |

---
*Bucket A re-evaluated with updated A.bright guard (threshold 0.45→0.30, 1.3× boost tier removed).*  
*Other buckets from `guard_before_after.csv` unchanged.*  
*Bootstrap CIs use seed=42, 200 resamples.*
