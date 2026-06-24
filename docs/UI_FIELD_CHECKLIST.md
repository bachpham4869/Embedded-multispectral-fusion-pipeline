# SmartBinocular — UI Field Checklist

Operator reference for the on-screen control chrome (P1/P2 features).
Tick each item during RPi4 field validation before considering the UI shippable for thesis demo.

---

## Button bar map

| Button | Key | Action |
|--------|-----|--------|
| `OPT`  | `1` | Switch to NIR (IMX290) single-sensor mode |
| `THM`  | `2` | Switch to thermal (MI48) single-sensor mode |
| `FUS`  | `3` | Switch to fused NIR+thermal mode |
| `CAP`  | `s` | Save still PNG from L1 raster (no chrome in saved file) |
| `REC`  | `v` | Toggle video recording on/off |
| `PROF` | `p` | Cycle processing profile: QUALITY → THROUGHPUT → RAW → QUALITY |

Additional keyboard controls (no on-screen button):
- `r` / `R` — toggle per-mode raw preview (no-op when profile=RAW)
- `e` / `E` — toggle E1/MAD detector on/off
- `+` / `-` — adjust fusion blend alpha (thermal layer weight)
- `a` — start 5-second auto-capture countdown

---

## Profile semantics

| Profile | Badge | What changes |
|---------|-------|--------------|
| QUALITY | `PROFILE: QUALITY` | Full pipeline — all enhancement stages active |
| THROUGHPUT | `PROFILE: THROUGHPUT` | Matches `RPI_THROUGHPUT_MAX_DEFAULTS` (see `config.py`); expect measurable FPS gain on RPi |
| RAW | `PROFILE: RAW [UNVERIFIED]` | No CLAHE, no HybridNIR, no LAB grading; closest to per-sensor capture. Thermal colormap step retained for visualization |

Profile badge shows `[UNVERIFIED]` when `display_profile_verified[profile]=False` in config.
The `raw` profile is unverified by default — set to True in `config.py` after RPi measurement confirms behaviour.

---

## RPi4 field validation checklist

### Button bar

- [ ] All six buttons (`OPT THM FUS CAP REC PROF`) reachable with gloved finger in <2 taps
- [ ] No accidental A1 probe movement when targeting the bottom bar
- [ ] Mode buttons (`OPT`/`THM`/`FUS`) produce visible mode change within 1–2 frames
- [ ] `CAP` button saves a PNG to `captures/` directory — verify chrome bar is absent in the file
- [ ] `REC` button starts/stops video; `[●REC]` indicator visible while recording
- [ ] `PROF` button cycles badge from `QUALITY → THROUGHPUT → RAW → QUALITY`
- [ ] Chrome fades to dim (α=0.15) after 5 s of no input; any click restores full brightness

### Still capture

- [ ] Saved PNG does NOT contain the soft-button bar (visual diff against live screen)
- [ ] Sidecar JSON contains `display_profile` field matching the active profile
- [ ] Sidecar JSON does NOT contain `/Users/<name>/` or `/home/<name>/` paths
- [ ] Sidecar JSON does NOT contain `_`-prefixed keys

### Video recording

- [ ] 60 s recording at each profile produces a playable `.mp4` on Mac (VLC/QuickTime)
- [ ] Stable framerate visible in HUD during recording (FPS counter should not drop >5 FPS vs idle)
- [ ] `dropped_count` shown in HUD when drops occur; reported in video sidecar `.meta.json`
- [ ] Disk limit hit (`capture_video_max_mb`) auto-stops recording with `REC STOPPED: DISK` toast

### Profile cycle

- [ ] `QUALITY → THROUGHPUT`: measurable FPS increase on RPi (measure and record delta below)
  - Measured FPS QUALITY: _____ fps
  - Measured FPS THROUGHPUT: _____ fps
  - Delta: _____ fps
- [ ] `QUALITY → RAW`: NIR enhancement artifacts visibly absent (compare side-by-side)
- [ ] `THROUGHPUT → RAW` and `RAW → QUALITY` transitions produce HUD toast within 1 frame
- [ ] Rate-limit between swaps (~1.5 s) prevents accidental rapid cycling
- [ ] `apply_profile("throughput")` regression test passes: `pytest tests/test_profile_apply.py`

### RAW + fusion interaction

- [ ] Switching to `RAW` while in fusion mode **immediately** switches display to single-sensor (IMX)
- [ ] HUD shows toast `RAW: single sensor only` for ~3 s
- [ ] After forced single-sensor, pressing `[FUS]` (or key `3`) **returns to fusion** — no auto-restore
- [ ] `r/R` key is a no-op while profile=RAW (no raw-preview toggle when already in raw profile)

### HUD badge

- [ ] Badge reads `PROFILE: QUALITY` in default config
- [ ] Badge reads `PROFILE: THROUGHPUT` after first `p` press
- [ ] Badge reads `PROFILE: RAW [UNVERIFIED]` (unverified suffix until confirmed on RPi)
- [ ] Badge updates within 1 frame of key press

---

## P2.5 status

Network-derived coarse position: **deferred — not enabled in this iteration.**
Add implementation note here if/when the feature is revisited.

---

## Notes / observed deltas (fill in during RPi session)

```
Date:
Hardware: RPi4B @ ____ GHz / ____ MB RAM
Firmware: ____

FPS measurements (3-mode × 3-profile grid):
  QUALITY   | imx: ___ | thermal: ___ | fusion: ___
  THROUGHPUT| imx: ___ | thermal: ___ | fusion: ___
  RAW       | imx: ___ | thermal: ___ | fusion: ___

Glove usability: pass / fail — notes: ____
Video playback: pass / fail — codec: mp4v @ ___ fps

Issues found:
  -
```
