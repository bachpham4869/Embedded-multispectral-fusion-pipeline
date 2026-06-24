# Duplicate Cluster Summary (conservative)

Clusters are duplicate-screening groups, not proof that every member is the same image.

| Metric | Value |
| --- | --- |
| mode | conservative |
| dhash_threshold | 4 |
| row_count | 14094 |
| dhash_coverage_rows | 5791 |
| file_sha256_coverage_rows | 5791 |
| cluster_count | 13375 |
| non_singleton_cluster_count | 255 |
| max_cluster_size | 333 |
| edge_count | 4447 |
| exact_file_sha_edges | 70 |
| dhash_edges | 4377 |

## Edge Category Counts

| Category | Count |
| --- | --- |
| dhash_false_positive_candidate | 2585 |
| exact_file_duplicate | 70 |
| likely_near_duplicate | 1792 |

## Cluster Size Distribution

| Cluster size | Cluster count |
| --- | --- |
| 1 | 13120 |
| 2 | 208 |
| 3 | 29 |
| 4 | 7 |
| 5 | 1 |
| 6 | 1 |
| 7 | 1 |
| 8 | 1 |
| 10 | 3 |
| 14 | 1 |
| 15 | 1 |
| 25 | 1 |
| 333 | 1 |

## Largest Clusters

| Cluster | Size | Labels | Sources |
| --- | --- | --- | --- |
| dupcluster::conservative::d049e4468382357c42e7a81b | 333 | {"fog": 113, "normal_day": 51, "rain": 44, "transition": 125} | {"offline_mwd": 155, "offline_weather11": 129, "offline_weather_time": 49} |
| dupcluster::conservative::24e854a23120d015cc3a39ad | 25 | {"fog": 11, "normal_day": 8, "transition": 6} | {"offline_mwd": 10, "offline_weather11": 11, "offline_weather_time": 4} |
| dupcluster::conservative::eaf49112055290482286a6c5 | 15 | {"normal_day": 15} | {"offline_weather_time": 15} |
| dupcluster::conservative::79597994a1e1f7fd6fbc084d | 14 | {"fog": 14} | {"offline_weather11": 14} |
| dupcluster::conservative::13ac967184d3e7f94dabcff8 | 10 | {"transition": 10} | {"offline_weather_time": 10} |
| dupcluster::conservative::8058f421d6a30de9193c67e0 | 10 | {"normal_day": 10} | {"offline_weather_time": 10} |
| dupcluster::conservative::e6c39b21bdc89bb0aa5fa6a5 | 10 | {"rain": 10} | {"offline_weather_time": 10} |
| dupcluster::conservative::24f360f9bfab561e3197632f | 8 | {"normal_day": 8} | {"offline_weather_time": 8} |
| dupcluster::conservative::537844dd429827837e42da51 | 7 | {"rain": 7} | {"offline_weather_time": 7} |
| dupcluster::conservative::27cef95899bfde5d7f143824 | 6 | {"normal_day": 6} | {"offline_weather_time": 6} |
| dupcluster::conservative::3427aa748722cc34f28690c8 | 5 | {"fog": 5} | {"offline_weather11": 5} |
| dupcluster::conservative::9434b79d9c1ccf3717f3d7cb | 4 | {"fog": 4} | {"offline_weather11": 4} |
| dupcluster::conservative::bfc56e1a68094e4a632b1d83 | 4 | {"normal_day": 4} | {"offline_weather_time": 4} |
| dupcluster::conservative::f73618b15b02110da30463c3 | 4 | {"normal_day": 4} | {"offline_mwd": 4} |
| dupcluster::conservative::93b11e46f316a39f52bc5381 | 4 | {"normal_day": 4} | {"offline_weather_time": 4} |
| dupcluster::conservative::44cfa63c2bfffe30d17915c4 | 4 | {"fog": 4} | {"offline_weather11": 4} |
| dupcluster::conservative::ad4c20596d9e7fa37d029681 | 4 | {"normal_day": 4} | {"offline_weather_time": 4} |
| dupcluster::conservative::00e45e77b118e9724e73187a | 4 | {"fog": 4} | {"offline_weather11": 4} |
| dupcluster::conservative::46550cf22a4eb505dd787b3f | 3 | {"normal_day": 3} | {"offline_mwd": 3} |
| dupcluster::conservative::1590d519e3c63569ea74d263 | 3 | {"fog": 3} | {"offline_weather11": 3} |
