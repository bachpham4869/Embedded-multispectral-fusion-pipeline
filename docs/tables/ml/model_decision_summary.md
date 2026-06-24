# Model Decision Summary

Status: Phase 1 freeze. No production model was retrained or overwritten.

| Candidate | Evidence | Pros | Cons | Decision | Migration gate |
| --- | --- | --- | --- | --- | --- |
| RF200 current config | Cluster-aware offline benchmark: accuracy 0.8263, balanced accuracy 0.7463, macro-F1 0.7362 | Best current tree baseline; familiar production path | Larger than RF100; still RGB-proxy/offline; no RPi4 timing | Current accuracy baseline | User confirmation plus sensor labels and RPi4 latency |
| RF100 | Cluster-aware offline benchmark: accuracy 0.8230, balanced accuracy 0.7415, macro-F1 0.7325 | Very close to RF200; smaller model artifact | Slight metric drop; deployment benefit not proven on RPi4 | Lightweight tree candidate | RPi4 feature+predict latency and user approval |
| ExtraTrees | Balanced accuracy 0.7386, macro-F1 0.7149 | Competitive balanced accuracy | Larger artifact; weaker macro-F1 than RF200 | Not selected | Reconsider only with sensor labels |
| HistGradientBoosting | Accuracy 0.8282, balanced accuracy 0.7096, macro-F1 0.7238 | Strong accuracy and compact artifact | Worse balanced accuracy; slower macOS proxy latency | Not selected for current production | Needs class-balanced improvement |
| MLP family | Best MLP variants remain >3 macro/balanced points below RF200 | Very small and fast proxy inference | Weak class-balanced performance; rejected by rule | Rejected for current production | Reopen only if new MLP closes balanced/macro gap with calibrated sensor labels |
| `optical_v2_candidate` | 4-class subset shows modest balanced/macro improvement | Better subset signal; relevant feature ideas | Only 4/9 classes; schema not production-versioned | Research candidate | Full 9-class coverage, latency, schema versioning, user approval |
| still-compatible 21-derived candidate | Same 4-class subset; temporal-only fields excluded, actual count 20 | Documents feature direction | Not the full temporal 21-feature set; subset-only | Research candidate | Same as v2 plus clear temporal/non-temporal split |
