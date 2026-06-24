# models/ei — Edge Impulse FOMO (person in dark)

Production default path: **`person_in_dark_fomo_int8.tflite`** (see `src/smartbinocular/config.py` → `ei_person.tflite_path`).

## Obtain the model

1. Export from Edge Impulse Studio: **Deployment → Linux → C++ library** (non-EON), or use your existing `person_in_dark-cpp-linux-v3-impulse-#2/` bundle.
2. Copy the INT8 model file from `…/tflite-model/tflite_learn_974465_7.tflite` (or equivalent) into this directory as:

   ```bash
   cp /path/to/export/tflite-model/tflite_learn_974465_7.tflite \
      models/ei/person_in_dark_fomo_int8.tflite
   ```

3. `*.tflite` in this folder is **gitignored** (large binary).

## Overrides

- **Env:** `EI_PERSON_TFLITE_PATH=/absolute/or/relative/path/model.tflite`
- **Config:** `ei_person.tflite_path` in `CONFIG`

## Smoke test

```bash
pip install tflite-runtime
python tools/ei_smoke.py
```

## Live pipeline

Enable with `EI_PERSON_IN_DARK_ENABLED=1`. The worker receives the **same BGR frame as the on-screen preview** for the current mode (imx / thermal / fusion) and display profile, **after** grade/cap and light drawing, **before** L1 HUD text.
