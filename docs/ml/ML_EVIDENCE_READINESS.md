# ML Evidence Readiness

Status: Phase 1 evidence freeze for Agent A4. This document separates
thesis-usable offline evidence, caveated proxy evidence, and items that remain
not measured.

## Readiness Summary

| Category | Current status | Report usage |
| --- | --- | --- |
| Offline duplicate-cluster-aware benchmark | caveated | Usable as the current best offline `optical_12_baseline` evidence. It is not source-held-out and not live NIR/LWIR validation. |
| Model selection | caveated | RF200 remains the accuracy baseline; RF100 is the lightweight tree candidate; MLP is rejected for current production. |
| Feature v2 / 21-derived candidates | preliminary | Research-only because the fair comparison is limited to a 4-class verified subset. |
| Raw sensor domain shift | preliminary | Use for drift, confidence, abstention, and labeling priority. Do not report accuracy. |
| Paired NIR/thermal evidence | preliminary | Use for paired sensor drift and behavior. It is unlabeled and uses RGB-scaler proxy inference. |
| Agent/manual-reviewed sensor subset | preliminary | Can support a limitation statement, not a final sensor-real metric. |
| User-confirmed sensor labels | not measured | Required before final sensor accuracy/F1 claims. |
| RPi4 latency | not measured | Required before claiming the 20 FPS deployment target. |

## Frozen Evidence To Carry Into Report

- The current best offline split is the duplicate-cluster-aware group split:
  `docs/tables/ml/cluster_aware_leakage_summary.md` reports zero exact JSON
  row, file SHA, split-group, duplicate-cluster, and dHash cross-split overlap
  within available coverage. Source-name overlap remains, so the split must
  not be called source-held-out.
- The current same-feature benchmark is
  `docs/tables/ml/model_comparison_cluster_aware.md`. RF200 has accuracy
  0.8263, balanced accuracy 0.7463, and macro-F1 0.7362. RF100 is close
  (accuracy 0.8230, balanced accuracy 0.7415, macro-F1 0.7325) and roughly
  halves model size in the proxy artifact.
- The MLP family remains below RF200 on balanced/macro metrics by more than
  the production decision threshold. It should not replace the tree baseline
  without new labeled sensor evidence.
- The raw sensor and paired NIR/thermal evaluations are behavior/domain-shift
  evidence only. They are not accuracy evidence because user-confirmed labels
  are absent.
- The agent-labeled subset is explicitly marked `agent_manual_label`. It is
  useful for accelerated review and for demonstrating likely RGB-proxy mismatch
  on sensor imagery, but user confirmation is still needed before final thesis
  performance claims.

## Output Matrix

See `docs/tables/ml/ml_evidence_readiness_matrix.md`.

## Remaining Gates

1. User-confirm the raw/paired label templates or provide completed manual
   labels.
2. Run the RPi4 latency protocol for RF100/RF200 and feature extraction.
3. Keep feature v2/21 as research-only until all 9 ENV classes are covered and
   schema versioning/backward-compatible loading exists.
4. Do not migrate RF100, v2, or 21-derived features into production without
   explicit confirmation.
