# Dataset Candidate Summary

Status: metadata-only. No candidate dataset was downloaded or mixed into
training/testing.

| Dataset | Link/source | License/usage constraints | Size estimate | Modality | Classes/labels available | Proposed ENV mapping | Primary gap | Risk | Recommendation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BDD100K | https://github.com/ucbdrive/bdd100k/blob/master/doc/format.md | Official download terms required | large, 100K images/video clips | RGB video/keyframe | weather, scene, timeofday | day/night/dawn-dusk/rain/fog | source-diverse weather/time | driving-only, no NIR/thermal | ask before download |
| ACDC | https://arxiv.org/abs/2104.13395 | official terms required | 4,006 labeled adverse images / 8,012 total reported in paper context | RGB | fog, night, rain, snow + segmentation | fog/rain/night | adverse validation | segmentation not ENV classifier | metadata only |
| DAWN | https://arxiv.org/abs/2008.05402 | original terms unclear; Roboflow mirror CC BY 4.0 | about 1,000 images in literature | RGB | detection under fog/rain/snow/sandstorm | fog/rain only | severe weather | detection labels and mirror risk | verify before use |
| RESIDE | https://hyper.ai/en/datasets/18179 | listed Other; official terms needed | 43.21 GB archive listed | RGB | synthetic/real haze subsets | fog/haze proxy | fog features | huge, synthetic bias | reject large download now |
| ExDark | https://github.com/cs-chan/Exclusively-Dark-Image-Dataset | repo terms must be checked | medium | RGB low-light | object classes under low light | night proxy | low-light diversity | no ENV/weather labels | metadata only |
| GLARE | https://github.com/NicholasCG/GLARE_Dataset | repo terms must be checked | unknown | RGB driving | glare-related labels | glare/backlight review | low-support glare | task mismatch | ask before small download |
| RGB-NIR Scene | https://www.epfl.ch/labs/ivrl/research/downloads/rgb-nir-scene-dataset/ | academic terms review needed | 1 GB full, 36 MB browser | aligned RGB/NIR | 9 scene categories | NIR domain only | sensor modality | no ENV labels | metadata only |
| LLVIP | https://github.com/bupt-ai-cz/LLVIP | citation/terms required | large, over 30K paired images reported by project summaries | visible/IR | pedestrian/fusion labels | low-light IR proxy | low-light IR | not active NIR ENV | metadata only |
| FLIR ADAS | https://www.flir.co.uk/oem/adas/dataset/ | free with terms/form | starter about 14K images; enhanced sets larger | thermal / ADAS | object labels and weather by subset | thermal domain only | thermal validation | object labels, terms gate | ask before download |
| PST900 | https://github.com/ShreyasSkandanS/pst900_thermal_rgb | GPL/terms must be verified | about 6 GB reported by dataset index | RGB-T-depth | segmentation/object classes | thermal fusion only | aligned RGB-T | subterranean domain | metadata only |
| RoadScene | https://github.com/sudao-he/RoadScene | repo terms must be checked | unknown | visible/IR | image fusion scenes | fusion reference only | visible/IR fusion | weak ENV labels | metadata only |
