# Manifest v2 Provenance Notes

**File:** `data/eval/nir_val/manifest_v2.csv`  
**Created:** 2026-04-28  
**Tool:** `tools/build_manifest_v2.py`  
**Seed:** 42  
**Commit at creation:** f1a3c48d254c068b57f31c69f88dfd5aee1b99b1

## Per-Class Quotas

| env_class | n | sources |
|-----------|---|---------|
| night_clear | 40 | `data/weather/darkface/image/` (extreme_dark subset, mean_L < 30) |
| normal_night | 40 | `data/weather/ExDark/` (Bicycle, Boat, Cup, Dog, Cat categories) |
| nir_night | 40 | `data/weather/gray/` (mid-brightness 60–110 range) |
| fog | 30 | `data/weather/weather11/dataset/fogsmog/` |
| glare | 30 | `data/weather/glare/real/input/` (pct_sat > 0.05 preselect) |
| backlight | 25 | `data/weather/backlight/` |
| rain | 20 | `data/weather/weather11/dataset/rain/` (still-only; no temporal stack) |
| normal_day | 20 | `data/weather/weather11/dataset/` (dew, frost, rainbow daylight) |
| mixed_edge | 25 | hand-picked borderline scenes: dark+glare, foggy+night, partial blowout |
| **Total** | **270** | 9 env classes |

## Edge Case Tags (column: `edge_case`)

| Tag | Criterion | Count |
|-----|-----------|-------|
| `extreme_dark` | mean_L < 15 | 37 |
| `mixed_lighting` | hand-labeled borderline | 25 |
| (blank) | no special edge case | 208 |

## Schema

```
path, env_class, edge_case, source_dir, mean_L, pct_sat, sha256
```

- `path`: absolute path to the image
- `env_class`: one of the 9 ENV_CLASS labels (production dispatch target)
- `edge_case`: optional tag for stress-test conditions
- `source_dir`: original source directory (for reproducibility)
- `mean_L`: mean of the L channel in LAB colorspace (0–255)
- `pct_sat`: fraction of pixels with L ≥ 250 (saturation check)
- `sha256`: first 16 hex chars of image file SHA-256 (identity check)

## Exclusions

- `from_logs_test.jsonl` is OFF-LIMITS for Phase 3 (ML eval only, Phase 4).
- Still images only; Bucket E (rain temporal median) requires frame sequences.
  Rain class images are included but evaluated with `still_image_cold_start_mode`.
- Bucket F evaluation uses `nir_b_ema_norm = mean(green)/255` per still as a
  single-frame proxy for the live EMA (documented limitation).

## Reproducibility

To regenerate this manifest from scratch:
```bash
python tools/build_manifest_v2.py \
    --out data/eval/nir_val/manifest_v2.csv \
    --seed 42
```
