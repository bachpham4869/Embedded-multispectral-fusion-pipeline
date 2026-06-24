# Auto-Labeling Model Candidates

Status: no independent teacher is used in the current phase. RF/heuristic output is `suggested_label` only.

| Method | Availability | Label type | Mapping | Cost | Limitations | Decision |
| --- | --- | --- | --- | --- | --- | --- |
| clip_zero_shot | unavailable | auto_weak_label only if enabled in a separate approved plan | medium if local CLIP/OpenCLIP weights exist | medium/high on CPU | not installed locally in current audit; no downloads in this phase | deferred; independent teacher unavailable locally |
| torchvision_pretrained | unavailable | auto_weak_label only if a local model is approved | low/medium; ImageNet labels do not map directly to ENV classes | medium on CPU | no direct ENV taxonomy output; no downloads in this phase | deferred; no local torchvision stack |
| timm_pretrained | unavailable | auto_weak_label only after separate approval | medium if Places/weather model weights are local | medium/high on CPU | not installed locally in current audit; no downloads in this phase | deferred; no local timm stack |
| onnx_tflite_local_classifier | unavailable | auto_weak_label only after model provenance is documented | depends on model labels | low/medium if edge model is small | no local runtime/model found in current audit | deferred; no local runtime/model available |
| external_vlm_api | not configured | auto_weak_label only after separate approved protocol | high with prompt mapping, but external and paid/remote | external API cost; latency depends on provider | not called without explicit approval | deferred; user approval required |
| rf_heuristic_suggested_label | available | suggested_label | high for suggestions because it already emits ENV labels | low | RF and heuristics share feature signals; agreement is not independent evidence | selected for manual-review acceleration; not an auto weak label source |
