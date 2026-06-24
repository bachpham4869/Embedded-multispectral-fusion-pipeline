# Dataset Candidate Survey

Status: Phase 2 metadata-only survey. No large dataset was downloaded and no
candidate data was mixed into train/test. Every candidate requires user
confirmation before download or integration.

## Selection Criteria

Candidates are useful only if they can improve evidence for one of these gaps:

- adverse weather: fog, haze, rain, smog, snow, sandstorm
- glare or backlight/HDR
- day/night/dawn/dusk
- low-light/NIR/active IR
- RGB-T or visible-thermal sensor fusion

## Candidate Summary

| Dataset | Source | License/terms status | Modality | Label format | Candidate mapping | Expected benefit | Domain mismatch risk | Recommendation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BDD100K | [official labels/docs](https://github.com/ucbdrive/bdd100k/blob/master/doc/format.md) | Requires official download/terms review | RGB video/keyframes | weather, scene, timeofday plus detection/segmentation | `normal_day`, `normal_night`, `rain`, `fog`, `transition` via `dawn/dusk` | Large source-diverse weather/time labels | Driving scenes; no NIR/thermal; label noise in tags | metadata only, ask before download |
| ACDC | [paper/source page](https://arxiv.org/abs/2104.13395) | Terms must be reviewed on official dataset page | RGB driving images | condition split: fog/night/rain/snow; segmentation labels | `fog`, `rain`, `normal_night`; snow not current ENV class | Strong adverse-condition benchmark | Segmentation task, automotive domain, no glare/backlight | metadata only, ask before download |
| DAWN | [paper](https://arxiv.org/abs/2008.05402), [Roboflow mirror](https://universe.roboflow.com/adverse-weather-detection/dawn-rkp3e) | Roboflow mirror reports CC BY 4.0; original terms need confirmation | RGB images | object detection, adverse weather groups | `fog`, `rain`; sandstorm/snow may be out-of-taxonomy | Severe weather diversity | Detection labels, not ENV labels; source/mirror ambiguity | metadata only, verify source before download |
| RESIDE | [dataset summary](https://hyper.ai/en/datasets/18179) | Listed as Other; large 43.21 GB archive; official terms needed | RGB synthetic/real haze | haze/dehaze subsets | `fog`/haze proxy only | Haze/fog feature engineering | Synthetic bias; very large; not ENV classification | reject for Phase 2 download, metadata only |
| ExDark | [official GitHub](https://github.com/cs-chan/Exclusively-Dark-Image-Dataset) | Check repository terms before use | RGB low-light images | object categories under low-light conditions | `normal_night`, `night_clear` proxy | Low-light optical baseline support | Object classification, not ENV class; no NIR | metadata only |
| GLARE | [official GitHub](https://github.com/NicholasCG/GLARE_Dataset) | Repository license/terms must be checked before use | RGB driving images | glare-related traffic-sign/detection labels | `glare`; possible `backlight` review | Directly relevant to glare weakness | Task-specific labels; may not cover binocular scenes | ask before small metadata download only |
| RGB-NIR Scene | [EPFL dataset page](https://www.epfl.ch/labs/ivrl/research/downloads/rgb-nir-scene-dataset/) | Academic dataset page; terms need review | aligned RGB + NIR still images | 9 scene categories | NIR domain-shift/reference only; not ENV classes | Real NIR channel sanity check | Scene categories do not map to weather/night labels | metadata only, useful for NIR feature study |
| LLVIP | [official GitHub](https://github.com/bupt-ai-cz/LLVIP) | Citation and dataset terms on repository/site | visible + infrared paired low-light images | pedestrian detection/fusion | `normal_night`, `nir_night` proxy; RGB-IR fusion evidence | Low-light visible/IR paired data | Pedestrian task; IR may not match active NIR IMX290 | metadata only |
| FLIR ADAS Thermal | [official page](https://www.flir.co.uk/oem/adas/dataset/) | Free dataset but requires FLIR terms/form | thermal, some visible/thermal products | object detection labels; weather metadata varies by subset | thermal feature/domain-shift only | Real thermal ADAS evidence | Detection labels; sensor/domain mismatch; terms gate | ask before download |
| PST900 | [official GitHub](https://github.com/ShreyasSkandanS/pst900_thermal_rgb) | Dataset Ninja reports GNU GPL 3.0; verify repository | aligned RGB + thermal + depth | semantic segmentation/object classes | thermal fusion only; no ENV labels | Small aligned RGB-T reference | Subterranean robotics domain, not weather ENV | metadata only |
| RoadScene | [dataset GitHub](https://github.com/sudao-he/RoadScene) | Repository terms must be checked | visible + infrared road scenes | fusion benchmark images | RGB-IR fusion qualitative reference | Road-scene visible/IR fusion | Often no ENV labels; not aligned with RF features | metadata only |

## Mapping Rules Before Any Integration

- Do not map task labels directly to ENV classes without an auditable mapping
  file and support counts.
- Keep candidate data under `data/_candidate_datasets/<dataset>/` only.
- Require `SOURCE.md`, `mapping_proposal.yaml`, license note, size estimate,
  and SHA256 manifest before any training/test use.
- Ask user confirmation before any large download or before mixing data into
  train/test.

## Current Recommendation

Best candidates for Phase 3 metadata review:

1. BDD100K for weather/time-of-day tags and source diversity.
2. ACDC for fog/night/rain adverse-condition validation.
3. GLARE for direct glare/backlight support after license review.
4. RGB-NIR Scene or LLVIP for NIR/IR domain-shift study, not direct ENV
   classifier training.
5. FLIR ADAS only if thermal feature/fusion validation becomes in scope.
