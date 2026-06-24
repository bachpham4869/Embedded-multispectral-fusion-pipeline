# Prepared Captions And Labels

Status: A4 preparation. These captions are draft text for later LaTeX patching.

## ML Tables

| Asset | Caption draft | Label |
| --- | --- | --- |
| Offline model comparison | Offline optical RGB-proxy model comparison on the duplicate-cluster-aware split. Metrics are not live NIR/LWIR validation; latency values are from the offline evaluation platform and must not be used as target-hardware deployment proof. | `tab:ch6-ml-cluster-aware-models` |
| Model decisions | Model-selection decisions for the current evidence freeze. Migration gates require user-confirmed sensor labels and target-hardware latency where applicable. | `tab:ch6-ml-model-decisions` |
| Class decisions | ENV class decisions after offline and sensor-proxy evidence review. Sensor-domain rows are caveated because user-confirmed labels are not available. | `tab:ch6-ml-class-decisions` |
| Domain shift summary | Raw and paired sensor proxy inference summary. The table reports behavior and drift, not sensor-real accuracy. | `tab:ch6-ml-domain-shift-summary` |

## ML Figures

| Asset | Caption draft | Label |
| --- | --- | --- |
| RF200 cluster-aware confusion matrix | Confusion matrix for the RF200 offline RGB-proxy baseline on the duplicate-cluster-aware split. | `fig:ch6-ml-rf200-cluster-confusion` |
| Raw sensor confidence histogram | Confidence distribution for raw sensor proxy inference. The rows are unlabeled, so the histogram is not an accuracy result. | `fig:ch6-ml-raw-sensor-confidence` |
| Paired NIR PCA | Feature-space visualization for paired NIR proxy rows relative to the offline baseline feature distribution. | `fig:ch6-ml-paired-nir-pca` |

## Fusion Tables

| Asset | Caption draft | Label |
| --- | --- | --- |
| Fusion generated summary | Strict-paired generated fusion and image-processing evidence. Fusion rows use real paired inputs but generated offline outputs; they do not validate captured runtime fusion. | `tab:ch6-fusion-generated-summary` |
| Per-bucket evidence | Per-bucket evidence status on strict paired inputs. Forced offline algorithm metrics are not runtime bucket-selection performance. | `tab:ch6-fusion-bucket-evidence` |
| Alignment diagnostics | Pairing skew and alignment evidence gates for strict paired NIR and thermal-display rows. | `tab:ch6-fusion-alignment-diagnostics` |
| Limitations table | Current evidence gates that bound the ML and fusion claims. These gaps define future validation work rather than negative final results. | `tab:ch7-evidence-gates` |

## Fusion Figures

| Asset | Caption draft | Label |
| --- | --- | --- |
| Strict paired fusion comparison grid | Generated fusion candidate comparison on strict paired NIR and thermal-display inputs. The figure illustrates offline candidate behavior, not captured runtime fusion. | `fig:ch6-fusion-strict-paired-grid` |
| Strict paired failure cases grid | Diagnostic failure examples from generated fusion candidates. These examples support limitation analysis and are not proof of runtime failure. | `fig:ch7-fusion-failure-grid` |
