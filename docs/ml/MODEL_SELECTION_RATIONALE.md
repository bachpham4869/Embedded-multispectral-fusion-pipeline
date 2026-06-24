# Model Selection Rationale

Status: Phase 2 quick benchmark. All model-comparison metrics are
`quick / preliminary / not thesis-ready` because image-level duplicate checks
are unavailable and the full `--bootstrap 500 --latency-repeats 300` run was
stopped for runtime.

## Benchmark Scope

Direct comparison is limited to same-feature classical ML models using the
existing `optical_12_baseline` feature vector and current frozen JSONL split.
No new dataset was mixed in and no production model was retrained or
overwritten.

Command:

```bash
PYTHONUNBUFFERED=1 .venv/bin/python tools/compare_classifiers.py \
  --train data/training/from_logs_train.jsonl \
  --test data/training/from_logs_test.jsonl \
  --models logistic_regression linear_svm sgd_linear_svm gaussian_nb decision_tree random_forest_50 random_forest_100 random_forest_200_current_config random_forest_depth_8 random_forest_depth_12 random_forest_depth_none extra_trees gradient_boosting hist_gradient_boosting knn mlp_small_32 mlp_small_64_32 \
  --bootstrap 100 \
  --latency-repeats 100 \
  --hardware-label "macOS proxy benchmark (not Raspberry Pi 4 CPU)" \
  --persist-models \
  --quick
```

Run manifest:
`artifacts/ml/model_comparison_12features/run_manifest.json`

## Key Results

| Model | Accuracy | Balanced accuracy | Macro-F1 | Model size MB | Mean latency ms | p95 latency ms | Status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| random_forest_100 | 0.8476 | 0.7677 | 0.7661 | 21.88 | 14.654 | 14.620 | quick / preliminary |
| random_forest_200_current_config | 0.8471 | 0.7665 | 0.7632 | 43.94 | 30.460 | 58.415 | quick / preliminary |
| random_forest_depth_none | 0.8457 | 0.7639 | 0.7615 | 44.96 | 43.812 | 129.621 | quick / preliminary |
| random_forest_50 | 0.8476 | 0.7638 | 0.7609 | 10.97 | 13.996 | 14.494 | quick / preliminary |
| extra_trees | 0.8088 | 0.7487 | 0.7262 | 37.81 | 39.805 | 82.460 | quick / preliminary |
| random_forest_depth_12 | 0.8202 | 0.7475 | 0.7367 | 25.25 | 25.721 | 29.531 | quick / preliminary |
| hist_gradient_boosting | 0.8443 | 0.7300 | 0.7471 | 1.93 | 88.459 | 204.187 | quick / preliminary |
| decision_tree | 0.7582 | 0.6849 | 0.6572 | 0.27 | 0.045 | 0.057 | quick / preliminary |
| logistic_regression | 0.6872 | 0.6439 | 0.6073 | 0.002 | 0.049 | 0.063 | quick / preliminary |
| gaussian_nb | 0.6559 | 0.5549 | 0.5487 | 0.002 | 0.198 | 0.238 | quick / preliminary |

Full details are in `docs/tables/ml/model_comparison_12features.md`,
`docs/tables/ml/model_comparison_12features.csv`, and
`docs/tables/ml/model_comparison_12features_ci.md`.

## RF 200 Current Config

Advantages:

- Best family in the quick same-feature benchmark, with balanced accuracy close
  to RF 100.
- Handles nonlinear thresholds and interactions in the handcrafted optical
  features better than linear/NB baselines.
- Existing production loader already supports the RF bundle and feature schema.

Disadvantages:

- Model artifact is about 43.94 MB for the benchmark RF 200 variant.
- Proxy single-sample latency is higher than RF 50/100 and not validated on
  Raspberry Pi 4 CPU.
- The frozen production model's held-out score remains inconsistent with
  sidecar TimeSeriesSplit CV, so this is not thesis-ready evidence.

## Lighter Alternatives

RF 50 and RF 100 are the most credible lighter same-feature alternatives in the
quick run. RF 100 slightly leads balanced accuracy and is roughly half the size
of RF 200. RF 50 is much smaller and close in aggregate metrics, but low-support
classes still need confidence intervals and source-diversity review.

Decision tree, logistic regression, Linear SVM/SGD, and GaussianNB are much
lighter but lose too much macro-F1/balanced accuracy for the current class mix.
They are useful as lower-bound baselines, not as replacements yet.

## Heavier Or Less Practical Alternatives

ExtraTrees and depth-unlimited RF are competitive but do not clearly beat RF
100/200 under this split and can have worse artifact size or latency. Histogram
gradient boosting has good accuracy but weak balanced accuracy and high proxy
latency in this run.

KNN and small MLP remain exploratory. They are not recommended for deployment
without further RPi4 measurement and calibration checks.

## Recommendation

Current engineering recommendation: keep the production schema/model unchanged
for now, treat RF 100 as a serious candidate for a future migration, and do not
replace RF 200 until a source-aware or leakage-clean split plus RPi4 CPU latency
benchmark confirms the tradeoff.

Migration plan if RF 100 is later selected:

1. Rebuild the train/test split with image/session/source metadata.
2. Run source-aware CV and frozen held-out evaluation with full provenance.
3. Run RPi4 CPU latency at the system target of 20 FPS.
4. Save the new model under a non-production path first.
5. Add sidecar versioning and SHA256 registry entry.
6. Switch production only after user confirmation.

Conclusion: not thesis-ready yet. Use this as preliminary engineering evidence
and as the basis for the next benchmark on a leakage-clean/source-aware split.

## Phase 3 Group-Aware Benchmark

Phase 3 reran focused same-feature benchmarks on
`data/training_v2/group_aware_train.jsonl` and
`data/training_v2/group_aware_test.jsonl`.

Status: `preliminary / not thesis-ready` because dHash near-duplicate screening
still finds 1,313 cross-split pairs and all latency is macOS proxy latency.

| Model | Accuracy | Balanced accuracy | Macro-F1 | Mean latency ms | P95 latency ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| random_forest_100 | 0.8363 | 0.7618 | 0.7533 | 14.966 | 16.282 |
| random_forest_200_current_config | 0.8358 | 0.7564 | 0.7502 | 14.913 | 16.267 |
| extra_trees | 0.8017 | 0.7285 | 0.7039 | 15.131 | 16.291 |
| logistic_regression | 0.6763 | 0.6419 | 0.6010 | 0.028 | 0.035 |
| linear_svm | 0.7194 | 0.6043 | 0.5786 | 0.030 | 0.037 |
| hist_gradient_boosting | 0.8490 | 0.7404 | 0.7573 | 25.379 | 38.710 |
| mlp_small_32 | 0.7903 | 0.6550 | 0.6744 | 0.060 | 0.062 |

Interpretation:

- RF100 has the strongest balanced accuracy in the focused group-aware run.
- RF200 is very close but roughly doubles serialized model size versus RF100.
- HistGradientBoosting has the highest accuracy and macro-F1 here, but lower
  balanced accuracy and higher proxy latency than RF100/RF200.
- Logistic regression and Linear SVM remain useful lightweight lower-bound
  baselines, not replacements.

Recommendation remains conservative: do not replace the production RF200 bundle
yet. RF100 is still the best lightweight migration candidate once dHash/source
overlap and RPi4 CPU latency are resolved.

## Phase 4 Duplicate-Cluster-Aware Benchmark

Phase 4 reran focused same-feature benchmarks on the conservative
duplicate-cluster-aware group split:

- train: `data/training_v3/cluster_aware_conservative_train.jsonl`
- test: `data/training_v3/cluster_aware_conservative_test.jsonl`
- feature set: `optical_12_baseline`
- leakage check: 0 exact row, file SHA, split-group, duplicate-cluster, and
  dHash cross-split pairs within available hash coverage
- source-name overlap: 8 sources, so this is not source-held-out
- latency: macOS proxy, not RPi4 CPU

| Model | Accuracy | Balanced accuracy | Macro-F1 | Mean latency ms | P95 latency ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| random_forest_100 | 0.8230 | 0.7415 | 0.7325 | 14.824 | 16.522 |
| random_forest_200_current_config | 0.8263 | 0.7463 | 0.7362 | 15.816 | 17.017 |
| extra_trees | 0.8008 | 0.7386 | 0.7149 | 15.211 | 16.285 |
| logistic_regression | 0.6673 | 0.6335 | 0.5927 | 0.027 | 0.033 |
| linear_svm | 0.7170 | 0.6108 | 0.5845 | 0.026 | 0.029 |
| hist_gradient_boosting | 0.8282 | 0.7096 | 0.7238 | 25.496 | 31.402 |
| mlp_small_32 | 0.7941 | 0.6614 | 0.6758 | 0.085 | 0.112 |

Interpretation:

- RF200 is the strongest focused model on balanced accuracy in Phase 4.
- RF100 remains close while using roughly half the serialized model size, so it
  remains the lighter migration candidate.
- HistGradientBoosting has the highest accuracy but weaker balanced accuracy and
  slower proxy latency, so it is not the first deployment candidate.
- Linear SVM and logistic regression are much smaller but lose substantial
  balanced accuracy.

Recommendation: keep the current production RF200 bundle until a deliberate
production migration is approved. RF100 is the best lightweight candidate for a
future non-production migration experiment, but Phase 4 does not retrain or
overwrite production.

Thesis wording: the Phase 4 metrics are acceptable as offline optical RGB-proxy
evidence with caveats. They are not live NIR/LWIR validation and not final RPi4
CPU latency evidence.

## Phase M5-M9 MLP and Raw Sensor Update

Raw sensor video evaluation added 590 sampled frames from
`data/raw_sensor_captures/test_30fps_morning.mp4`. These frames are unlabeled,
so they support domain-shift and uncertainty analysis only. Predictions are
reported as `RGB-scaler proxy inference` because the current production bundle
has an RGB scaler, not a validated NIR scaler.

MLP-family benchmark on the Phase 4 duplicate-cluster-aware split used seeds
42, 43, and 44. Calibrated MLP rows used an internal train/validation split;
the test set was not used for calibration.

| Model | Balanced accuracy mean | Macro-F1 mean | ECE mean | Latency mean ms | Decision |
| --- | ---: | ---: | ---: | ---: | --- |
| RF100 | 0.7477 | 0.7384 | 0.0897 | 13.9686 | lightweight tree candidate |
| RF200 current config | 0.7480 | 0.7380 | 0.0895 | 24.0612 | accuracy baseline |
| best MLP (`mlp_64_32`) | 0.6882 | 0.6973 | 0.0123 | 0.2686 | reject for production now |
| best calibrated MLP (`calibrated_mlp_64`) | 0.6382 | 0.6525 | 0.0412 | 1.0549 | reject for production now |

Decision: MLP variants are much faster and often better calibrated, but their
balanced accuracy and macro-F1 trail RF200 by more than 3 percentage points.
Under the agreed rule, reject MLP for current production. Keep RF200 as the
accuracy baseline and RF100 as the main lightweight candidate pending RPi4 CPU
latency validation.

Feature-set ablation found a small subset-only gain for `optical_v2_candidate`
on four verified-image classes (`fog`, `normal_day`, `rain`, `transition`):
RF200 balanced accuracy improved from 0.7482 to 0.7584 and macro-F1 from 0.7590
to 0.7719. This is not a full 9-class result. `optical_21_candidate` could not
be benchmarked supervised because `temporal_brightness_std` is not valid for
still-image train/test rows and was not zero-imputed. No production schema
migration is recommended in this phase.
latency evidence.

## Phase M10 Final Current Recommendation

Manual sensor labels are still absent: runtime validation found only
`artifacts/ml/sensor_domain_shift/manual_label_template.csv` with 120 rows and
0 completed `manual_label` entries. Sensor-labeled accuracy is therefore `not
measured`.

Tiered recommendation:

| Tier | Decision | Evidence | Migration status |
| --- | --- | --- | --- |
| RF200 | current accuracy baseline | strongest or tied tree baseline on Phase 4/M10 evidence | keep production unchanged |
| RF100 | lightweight tree candidate | near RF200 with smaller artifact and lower proxy latency | candidate only; needs RPi4 timing |
| MLP family | rejected for current production | balanced accuracy/macro-F1 remain more than 3 points below RF200 | do not migrate |
| `optical_v2_candidate` | research candidate | small 4-class subset gain only | no schema migration |
| `optical_21_candidate_still` | research candidate | actual count 20 after excluding temporal-only feature; same subset evidence as v2 | no schema migration |
| `optical_21_candidate_temporal` | deferred | requires labeled sequential data | no supervised still-image claim |

Production migration gates:

1. valid manual-labeled raw sensor subset or stronger target-domain evaluation,
2. Raspberry Pi 4 CPU latency for feature extraction, model inference,
   feature+predict, load time, model size, and memory,
3. schema versioning and backward-compatible loader for any feature-set change,
4. explicit user confirmation before writing production model or sidecar files.

Current answer to the model-selection question: keep RF200 as the baseline,
keep RF100 as the main lightweight candidate, reject MLP for current production,
and treat v2/21 features as research-only until domain labels and RPi4 timing
exist.

## Phase M11 Suggested Labels Do Not Change Model Selection

Phase M11 does not add an independent teacher model. The local environment does
not provide CLIP/OpenCLIP/timm/ONNX/TFLite/API teacher capability, so the only
implemented path is RF/heuristic `suggested_label` generation for manual review.

This evidence does not change production model selection:

| Item | Status | Decision impact |
| --- | --- | --- |
| `suggested_label` | RF/heuristic review aid | helps prioritize labeling only |
| `auto_weak_label` | not generated | no training/test use |
| `manual_label` | still absent | raw sensor accuracy remains not measured |

RF200 remains the current accuracy baseline, RF100 remains the lightweight tree
candidate, MLP remains rejected for current production, and v2/21 features
remain research-only. No production migration is justified by M11 alone.

## Phase P12 Paired NIR/Thermal ML Evidence

Paired data adds target-domain domain-shift evidence, not model-selection
accuracy evidence:

| Item | Value |
| --- | --- |
| paired rows | 584 |
| pairing tier | `frame_strict`: 584 |
| trusted labels | 0 |
| inference scope | `RGB-scaler proxy inference, not validated NIR classifier accuracy` |
| top-1 distribution | `normal_night`: 579, `transition`: 5 |
| abstention at tau1=0.62 | 0.0086 |

Decision impact:

- RF200 remains the current accuracy baseline.
- RF100 remains the lightweight tree candidate.
- MLP remains rejected for current production.
- `optical_v2_candidate` and `optical_21_candidate_still` remain research
  candidates only.
- Paired data is not used to select a model by accuracy because there are no
  trusted labels.

Production migration gates are unchanged: trusted paired/manual labels, RPi4
latency, schema/version compatibility for any feature-set change, and explicit
user confirmation.

## Phase 1 Evidence Freeze

The final ML freeze keeps the model recommendation tiered:

| Candidate | Current decision |
| --- | --- |
| RF200 current config | Current accuracy baseline. Cluster-aware offline metrics: accuracy 0.8263, balanced accuracy 0.7463, macro-F1 0.7362. |
| RF100 | Lightweight tree candidate. It is close to RF200 on the same split (accuracy 0.8230, balanced accuracy 0.7415, macro-F1 0.7325) and has a smaller artifact, but RPi4 latency is still not measured. |
| MLP family | Rejected for current production because balanced accuracy / macro-F1 remain more than 3 points below RF200 despite fast proxy latency. |
| `optical_v2_candidate` / still-compatible 21-derived features | Research-only. The fair comparison is limited to a 4-class verified subset and cannot justify a production schema change. |

The agent-labeled sensor subset is not used to select a production model by
accuracy. It is only preliminary evidence that RGB-scaler proxy inference can
misclassify visually backlit sensor scenes as night-like classes. Production
migration still requires user-confirmed labels, RPi4 latency,
schema/version compatibility, and explicit approval.

See `docs/tables/ml/model_decision_summary.md`.
