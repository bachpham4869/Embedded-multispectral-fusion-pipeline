# Model Research Survey

Status: Phase 2 survey. Direct quantitative comparison is limited to
same-feature classical ML. Image-input and specialized models are separate
baselines or literature comparison, not direct competitors to RF on 12
handcrafted features.

## A. Same-Feature Classical ML Baselines

These consume the same `optical_12_baseline` vector and current train/test
split, so they are directly comparable. Phase 2 ran a quick benchmark with
`--bootstrap 100 --latency-repeats 100`; full benchmark was stopped for runtime.

| Model | Input | Direct benchmark? | Phase 2 result summary | RPi4 CPU 20 FPS feasibility | Notes |
| --- | --- | --- | --- | --- | --- |
| Logistic Regression | 12 features | yes | balanced accuracy 0.6439 | high | lightweight but too weak |
| Linear SVM | 12 features | yes | balanced accuracy 0.6126 | high | no native probability |
| SGDClassifier | 12 features | yes | balanced accuracy 0.5799 | high | exploratory online-style baseline |
| GaussianNB | 12 features | yes | balanced accuracy 0.5549 | high | simplest probabilistic baseline |
| Decision Tree | 12 features | yes | balanced accuracy 0.6849 | high | interpretable, weaker than RF |
| RF 50/100/200 | 12 features | yes | best family; RF100 balanced accuracy 0.7677 | plausible, needs RPi4 validation | current strongest same-feature option |
| ExtraTrees | 12 features | yes | balanced accuracy 0.7487 | uncertain proxy latency | RF-like but not better |
| GradientBoosting/HGB | 12 features | yes | HGB accuracy high but balanced accuracy 0.7300 | uncertain proxy latency | not selected now |
| KNN | 12 features | yes, exploratory | balanced accuracy 0.6906 | not recommended without RPi4 test | prediction cost scales with data |
| Small MLP | 12 features | yes, exploratory | balanced accuracy 0.6497/0.6669 | plausible but weaker | needs calibration if used |

Artifacts:

- `docs/tables/ml/model_comparison_12features.md`
- `docs/tables/ml/model_comparison_12features_ci.md`
- `artifacts/ml/model_comparison_12features/run_manifest.json`

## B. Image-Input / TFLite Baselines

These use images rather than the 12 handcrafted feature vector. They require a
separate image-input split, preprocessing policy, and RPi4/TFLite benchmark.

| Candidate | Original task | Modality | Pretrained availability | Compute cost | NIR/thermal fit | RPi4 CPU 20 FPS fit | Benchmark feasible in project? | Role |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MobileNetV3 | mobile image classification | RGB image | common pretrained weights; [paper](https://arxiv.org/abs/1905.02244) | low-medium | needs NIR fine-tune; no thermal | plausible only after TFLite test | yes, separate baseline | image-input baseline |
| EfficientNet-Lite | mobile/TFLite image classification | RGB image | [TensorFlow Lite family](https://blog.tensorflow.org/2020/03/higher-accuracy-on-vision-models-with-efficientnet-lite.html) | variant-dependent | needs NIR fine-tune | plausible for small/int8 variants | yes, separate baseline | image-input baseline |
| ShuffleNet | efficient mobile CNN | RGB image | [paper](https://arxiv.org/abs/1707.01083) | low | needs NIR fine-tune | plausible | optional | edge image baseline |
| MobileOne | mobile backbone | RGB image | [Apple/paper page](https://machinelearning.apple.com/research/mobileone) | low on supported mobile hardware | needs NIR fine-tune | unknown on RPi4 CPU | optional | future baseline |
| ResNet18-lite | image classification | RGB image | common pretrained weights | medium | weak without fine-tune | uncertain | optional | heavier reference |
| Places365-style CNN | scene recognition | RGB scene image | [Places365 models](https://github.com/CSAILVision/places365) | medium-heavy | no native NIR/thermal | uncertain | optional | scene-prior reference |
| CLIP/embedding classifier | zero-shot/reference | RGB image + text | [CLIP](https://openai.com/research/clip) | heavy | weak for NIR/thermal | unlikely for 20 FPS CPU | offline only | literature/future work |

## C. Literature / Specialized Model Directions

These answer the reviewer question about real-world specialized methods. They
are literature comparison unless a matching dataset and input policy are added.

| Direction | Original task | Input modality | Dataset/pretrained availability | NIR/thermal fit | RPi4 CPU 20 FPS fit | Benchmark feasible now? | Use in thesis |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Adverse weather recognition | fog/rain/snow/sandstorm recognition/detection | RGB image/video | BDD100K, ACDC, DAWN-like sources | RGB proxy only | depends on backbone | no | literature comparison and dataset motivation |
| Day/night/dawn scene classification | time-of-day scene class | RGB image/video | BDD100K timeofday labels | useful for `transition` review | possible with lightweight model | no | argue `transition` needs stronger evidence |
| Low-light/NIR scene classification | low-light visible/IR tasks | RGB + IR/NIR | LLVIP, RGB-NIR Scene | closer to NIR/IR but not live IMX290 | uncertain | no | domain-shift/future work |
| Glare/backlight detection | highlight/HDR detection | RGB image/video | GLARE and task-specific datasets | relevant to current low-support classes | possible | no | motivates feature engineering/data collection |
| RGB-T / visible-thermal understanding | detection/segmentation/fusion | RGB + thermal/IR | FLIR ADAS, PST900, LLVIP-style data | relevant to thermal branch | usually heavier | no | future multimodal comparison |
| Lightweight edge classifier | mobile image classification | RGB image | MobileNet/ShuffleNet/MobileOne families | requires fine-tune | possible after TFLite test | no | deployability comparison |

Detailed table: `docs/tables/ml/specialized_model_survey.md`.

## Decision

For Phase 2, category A is the only direct quantitative benchmark. Categories B
and C should be described as image-input baselines or literature comparison. Do
not claim a CNN/TFLite model is worse or better than RF unless it is evaluated
with a defined image-input protocol and the same data split policy.
