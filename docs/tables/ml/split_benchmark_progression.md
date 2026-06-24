# Split and Benchmark Progression

All rows use `optical_12_baseline` handcrafted features. Latency is macOS proxy
latency, not Raspberry Pi 4 CPU deployment latency.

## Split Integrity Progression

| Phase | Split | Train rows | Test rows | Exact file SHA overlap | dHash cross-split pairs at <=4 | Source-name overlap | Status |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| Phase 2 | existing `data/training/from_logs_*` | 11,981 | 2,113 | unverified | unverified | likely | preliminary |
| Phase 3 | group-aware by `split_group_id` | 11,981 | 2,113 | 0 | 1,313 | 8 | preliminary |
| Phase 4 | conservative duplicate-cluster-aware group split | 11,981 | 2,113 | 0 | 0 | 8 | offline thesis-use candidate with caveats |

Phase 4 still is not source-held-out because source-name overlap remains.

## Focused RF Progression

| Phase | Model | Accuracy | Balanced accuracy | Macro-F1 | Key caveat |
| --- | --- | ---: | ---: | ---: | --- |
| Phase 2 | random_forest_100 | 0.8476 | 0.7677 | 0.7661 | current split unresolved; quick benchmark |
| Phase 2 | random_forest_200_current_config | 0.8471 | 0.7665 | 0.7632 | current split unresolved; quick benchmark |
| Phase 3 | random_forest_100 | 0.8363 | 0.7618 | 0.7533 | 1,313 dHash cross-split screening pairs |
| Phase 3 | random_forest_200_current_config | 0.8358 | 0.7564 | 0.7502 | 1,313 dHash cross-split screening pairs |
| Phase 4 | random_forest_100 | 0.8230 | 0.7415 | 0.7325 | source-name overlap and non-RPi latency |
| Phase 4 | random_forest_200_current_config | 0.8263 | 0.7463 | 0.7362 | source-name overlap and non-RPi latency |

Interpretation: Phase 4 removes the measured dHash cross-split blocker within
hash-covered records, and the RF metrics drop versus the unresolved split. This
is useful evidence that earlier splits were optimistic.
