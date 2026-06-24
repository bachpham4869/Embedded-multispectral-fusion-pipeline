# Specialized Model Survey

Status: Phase 2 literature/model survey. Category A models are directly
benchmarked on `optical_12_baseline`; categories B and C are not directly
comparable unless an image-input or multimodal benchmark is implemented.

| Group | Model/direction | Original task | Input modality | Dataset/pretrained availability | Compute cost | RPi4 CPU 20 FPS fit | NIR/thermal fit | Benchmark feasible now? | Role vs handcrafted RF |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A | Logistic Regression | tabular classification | 12 optical features | sklearn | very low | high | same as current proxy features | yes, completed quick run | lower-bound lightweight baseline |
| A | Linear SVM / SGD | tabular classification | 12 optical features | sklearn | very low | high | same as current proxy features | yes, completed quick run | lightweight baseline without native calibrated probabilities for LinearSVC |
| A | GaussianNB | tabular classification | 12 optical features | sklearn | very low | high | same as current proxy features | yes, completed quick run | simplest probabilistic baseline |
| A | Decision Tree | tabular classification | 12 optical features | sklearn | very low | high | same as current proxy features | yes, completed quick run | interpretable but weaker macro-F1 |
| A | Random Forest variants | tabular classification | 12 optical features | sklearn | medium | plausible, must measure on RPi4 | same as current proxy features | yes, completed quick run | current best same-feature family |
| A | ExtraTrees | tabular classification | 12 optical features | sklearn | medium | uncertain due proxy latency | same as current proxy features | yes, completed quick run | RF-like alternative |
| A | GradientBoosting / HistGradientBoosting | tabular classification | 12 optical features | sklearn | low-medium train/infer varies | uncertain from proxy latency | same as current proxy features | yes, completed quick run | possible but not better on balanced accuracy |
| A | KNN | tabular classification | 12 optical features | sklearn | prediction grows with data | exploratory only | same as current proxy features | yes, exploratory quick run | not recommended without RPi4 latency |
| A | Small MLP | tabular classification | 12 optical features | sklearn | low-medium | plausible but needs calibration | same as current proxy features | yes, exploratory quick run | nonlinear lightweight reference |
| B | MobileNetV3 | mobile image classification | RGB/image | paper/common pretrained weights | low-medium TFLite | plausible only after RPi4 TFLite test | NIR needs fine-tune; no thermal | not now | image-input baseline, not direct RF competitor |
| B | EfficientNet-Lite | mobile/TFLite image classification | RGB/image | TensorFlow Lite ecosystem | variant-dependent | plausible for small variants | NIR needs fine-tune; no thermal | not now | image-input baseline |
| B | ShuffleNet | efficient image classification | RGB/image | paper/pretrained variants | low | plausible | NIR needs fine-tune | not now | edge image baseline |
| B | MobileOne | mobile image backbone | RGB/image | paper/code available | low on supported mobile hardware | unknown on RPi4 CPU | NIR needs fine-tune | not now | future image-input baseline |
| B | ResNet18-lite | image classification | RGB/image | common pretrained weights | medium | uncertain for 20 FPS CPU | weak without fine-tune | not now | heavier reference only |
| B | Places365-style model | scene recognition | RGB scene image | Places365 models | medium-heavy | uncertain | no native NIR/thermal | not now | scene-prior literature baseline |
| B | CLIP/embedding classifier | zero-shot/reference | RGB image + text | pretrained CLIP | heavy | unlikely at 20 FPS CPU | weak NIR/thermal fit | no deploy benchmark | offline reference/future work |
| C | Adverse weather recognition CNN | weather recognition/classification | RGB image/video | weather datasets such as BDD/ACDC/DAWN-style | depends on backbone | only if lightweight | RGB proxy unless live NIR collected | literature only now | motivates image-input baseline |
| C | Day/night/dawn scene classifier | time-of-day scene class | RGB image/video | BDD timeofday and similar labels | low-medium | plausible with lightweight backbone | transition still needs live validation | literature only now | tests whether `transition` should be classifier state |
| C | Low-light/NIR scene classifier | low-light detection/fusion | visible + IR/NIR | LLVIP/RGB-NIR style datasets | medium | uncertain | closer to NIR/IR but not IMX290 active NIR | literature/candidate only | domain-shift reference |
| C | Glare/backlight detector | highlight/backlight detection | RGB image/video | specialized glare datasets | low if handcrafted, medium if CNN | plausible after data | no thermal required | not now | motivates feature v2 and live data |
| C | RGB-T / visible-thermal model | detection/segmentation/fusion | RGB + thermal/IR | FLIR, PST900, LLVIP-style data | medium-heavy | unlikely without optimized model | relevant to thermal branch | literature only now | future multimodal fusion baseline |
| C | Lightweight edge classifier | edge image classification | RGB/image | MobileNet/ShuffleNet/MobileOne families | low-medium | possible after TFLite benchmark | needs domain fine-tune | not now | deployability comparison |
