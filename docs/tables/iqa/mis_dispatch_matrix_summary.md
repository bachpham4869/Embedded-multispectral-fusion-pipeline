# Mis-Dispatch Matrix — 6×9 (Phase 3 v2, Round 0)

**Generated:** 2026-04-28  
**Source:** `docs/tables/iqa/raw/mis_dispatch_matrix.csv` (1620 rows, 270 images × 6 buckets)  
**Metrics:** delta values (after − before), bucket-induced only  

**Bold** = failure mode: Δcrush > 0.05 OR Δsat > 0.10 OR Δlog_rms < −0.30  
_Italic_ = near-passthrough: |Δrms| < 0.03 AND |Δcrush| < 0.005  
`✓` = correct dispatch per `OPTICAL_BUCKET_DISPATCH`

| Bucket | night_clea | normal_nig | nir_night | fog | glare | backlight | rain | normal_day | mixed_edge |
|--------|------------|------------|------------|------------|------------|------------|------------|------------|------------|
| A | Δrms=-0.12 crush=-0.677 sat=+0.058 ✓ | Δrms=-0.05 crush=-0.257 sat=+0.073 ✓ | **Δrms=+0.13 crush=-0.022 sat=+0.160** ✓ | Δrms=+0.04 crush=-0.002 sat=+0.021 | Δrms=-0.04 crush=-0.014 sat=+0.097 | **Δrms=-0.54 crush=-0.018 sat=+0.118** | Δrms=+0.00 crush=-0.011 sat=+0.098 | Δrms=-0.22 crush=-0.012 sat=+0.030 | **Δrms=-0.09 crush=-0.345 sat=+0.133** |
| B | Δrms=+0.10 crush=-0.675 sat=+0.000 | Δrms=+0.20 crush=-0.250 sat=-0.000 | Δrms=+0.21 crush=-0.016 sat=+0.001 | Δrms=+0.33 crush=-0.001 sat=-0.009 ✓ | Δrms=+0.21 crush=-0.013 sat=-0.003 ✓ | Δrms=+0.12 crush=-0.016 sat=-0.016 | Δrms=+0.24 crush=-0.008 sat=-0.006 | Δrms=+0.20 crush=-0.008 sat=+0.001 | Δrms=+0.18 crush=-0.342 sat=+0.000 |
| C | _Δrms=+0.01 crush=+0.000 sat=-0.000_ | Δrms=+0.04 crush=+0.000 sat=-0.004 | _Δrms=+0.02 crush=+0.000 sat=-0.010_ | _Δrms=+0.01 crush=+0.000 sat=-0.027_ | Δrms=+0.05 crush=+0.000 sat=-0.023 | Δrms=+0.03 crush=+0.000 sat=-0.077 ✓ | _Δrms=+0.02 crush=+0.000 sat=-0.022_ | _Δrms=-0.01 crush=+0.000 sat=-0.012_ ✓ | Δrms=+0.04 crush=+0.000 sat=-0.002 |
| D | **Δrms=-1.02 crush=+0.084 sat=-0.000** | **Δrms=-0.80 crush=+0.068 sat=-0.001** | Δrms=-0.26 crush=-0.002 sat=-0.003 | **Δrms=-0.39 crush=+0.001 sat=-0.011** | **Δrms=-0.67 crush=+0.046 sat=-0.010** | **Δrms=-0.96 crush=+0.041 sat=-0.037** | **Δrms=-0.58 crush=-0.001 sat=-0.014** | **Δrms=-0.75 crush=-0.002 sat=-0.005** | **Δrms=-0.98 crush=+0.122 sat=+0.000** |
| E | _Δrms=-0.01 crush=+0.000 sat=-0.000_ | _Δrms=-0.00 crush=+0.000 sat=-0.000_ | _Δrms=-0.01 crush=-0.001 sat=-0.000_ | _Δrms=-0.01 crush=-0.000 sat=-0.000_ | _Δrms=-0.01 crush=+0.000 sat=-0.000_ | _Δrms=-0.00 crush=-0.000 sat=-0.000_ | _Δrms=-0.01 crush=-0.000 sat=-0.000_ ✓ | _Δrms=-0.00 crush=-0.000 sat=-0.000_ | _Δrms=-0.01 crush=+0.002 sat=-0.000_ |
| F | Δrms=-0.11 crush=-0.677 sat=+0.049 | Δrms=-0.05 crush=-0.255 sat=+0.044 | Δrms=+0.04 crush=-0.014 sat=-0.010 | _Δrms=+0.01 crush=-0.000 sat=-0.026_ | Δrms=+0.02 crush=-0.014 sat=-0.017 | Δrms=-0.05 crush=-0.015 sat=-0.076 | Δrms=+0.03 crush=-0.008 sat=-0.022 | Δrms=-0.03 crush=-0.002 sat=-0.012 | **Δrms=-0.09 crush=-0.345 sat=+0.123** ✓ |

## Failure-mode catalog (bold cells)

| Bucket | env_class | Δlog_rms | Δcrush | Δsat | Note |
|--------|-----------|---------|--------|------|------|
| A | nir_night | +0.128 | -0.022 | +0.160 | oversaturation |
| A | backlight | -0.536 | -0.018 | +0.118 | oversaturation, large contrast loss |
| A | mixed_edge | -0.091 | -0.345 | +0.133 | oversaturation |
| D | night_clear | -1.023 | +0.084 | -0.000 | bucket adds crush, large contrast loss |
| D | normal_night | -0.804 | +0.068 | -0.001 | bucket adds crush, large contrast loss |
| D | fog | -0.388 | +0.001 | -0.011 | large contrast loss |
| D | glare | -0.666 | +0.046 | -0.010 | large contrast loss |
| D | backlight | -0.956 | +0.041 | -0.037 | large contrast loss |
| D | rain | -0.581 | -0.001 | -0.014 | large contrast loss |
| D | normal_day | -0.747 | -0.002 | -0.005 | large contrast loss |
| D | mixed_edge | -0.984 | +0.122 | +0.000 | bucket adds crush, large contrast loss |
| F | mixed_edge | -0.090 | -0.345 | +0.123 | oversaturation |