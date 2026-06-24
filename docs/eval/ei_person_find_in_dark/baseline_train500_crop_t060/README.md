# Baseline Eval Run — baseline_train500_crop_t060

> **Parity caveat (R4):** Results use sRGB JPEG images from the train split. The
> production pipeline feeds raw NIR frames. Harness results are upper bounds on
> serving accuracy only if the JPG distribution matches NIR statistics — it does not.
> See `DECISIONS_AND_RISKS.md R4`.

## Critical Finding — INT8 Softmax Cap

The `_fomo_postprocess` function (production + harness) applies softmax to the
dequantized INT8 logits. This model's output quantization parameters
(`scale=0.00390625, zero_point=-128`) compress the logit range to `[0.0, 0.9961]`.
**The theoretical maximum softmax person probability for this model is ~0.730**, computed
as `exp(0.9961) / (exp(0.9961) + exp(0.0)) ≈ 0.730`.

Consequence: the EI metadata threshold of `0.8` (`EI_CLASSIFIER_OBJECT_DETECTION_THRESHOLD`)
is **unreachable** at INT8 inference — it was calibrated on the float model where logits
span a wider range. **The production `config.py` threshold of `0.8` also produces zero
detections for this model on both NIR and JPG inputs.**

Effective threshold range for this INT8 model: `[0.0, 0.73)`. Canonical eval baseline
uses **threshold=0.60** as the primary working point; threshold=0.65 is included as a
conservative comparison.

Recommended follow-up: run device-side capture-and-replay on RPi4 at threshold=0.60
and update `config.py` `ei_person.threshold` accordingly.

---

## Epochs

| epoch | threshold | fit_mode | interp | centroid_hit F1 | image_f1 | P | R | tp | fp | fn |
|-------|-----------|----------|--------|-----------------|----------|---|---|----|----|----|
| epoch_00_raw_t060_crop_area | 0.60 | crop | area | **0.9701** | 0.9723 | 1.000 | 0.942 | 471 | 0 | 29 |
| epoch_00_raw_t065_crop_area ¹ | 0.65 | crop | area | 0.9572 | — | 1.000 | 0.918 | 459 | 0 | 41 |

¹ Run in sibling directory `../baseline_train500_crop_t065/`.

**Primary metric winner: epoch_00_raw_t060_crop_area (threshold=0.60)**

---

## Reproduce

```bash
pip install -e ".[ei]"

# ai-edge-litert works as tflite-runtime on Mac
pip install ai-edge-litert

python tools/eval_ei_person.py \
  --tflite "person_in_dark-cpp-linux-v3-impulse-#2/tflite-model/tflite_learn_974465_7.tflite" \
  --dataset data/find-person-in-the-dark/train/train \
  --pass raw --threshold 0.6 --fit-mode crop --interp area \
  --limit 500 --metric-primary centroid_hit \
  --run-id baseline_train500_crop_t060 \
  --out docs/eval/ei_person_find_in_dark/baseline_train500_crop_t060
```

Expected: `summary.json` with `centroid_hit.f1 ≈ 0.970`, `image_level_f1.f1 ≈ 0.972`.

---

## Dataset Notes

- **Split:** train only (`data/find-person-in-the-dark/train/train/`).
- **n_images evaluated:** 500 (first 500 by sorted stem). All 500 are GT-positive.
- **Degenerate boxes:** Dataset contains annotations with `xmin==xmax` (e.g., `020118.xml`).
  These are silently skipped per `discover.py` policy.
- **tflite SHA-256:** `0aea3da7c25ce0bbfeb67e007635bb60b6342abb07fb3c93d12c11c0d56797cc`
- **Git rev at run time:** `f9a4b1c9e8298ab25006bf01c5e70bb05da8bb72`

---

## Next Steps

1. **Threshold sweep** across `{0.50, 0.55, 0.60, 0.65, 0.70}` to confirm t=0.60 as
   the best working point. Stop condition: 3 consecutive epochs without improvement on
   centroid_hit F1.
2. **Fit-mode comparison** (`letterbox`, `passthrough`) at t=0.60 once baseline is
   confirmed — expected to differ slightly given Q4 finding.
3. **Larger subset** (`--limit 2000`) for stable CI on the chosen knob.
4. **Device verify:** capture NIR frames on RPi4, re-run with `--pass raw` to close
   the R4 distribution gap.
5. **Production threshold fix:** update `config.py` `ei_person.threshold` from `0.8`
   to `0.6` (pending device verify).
