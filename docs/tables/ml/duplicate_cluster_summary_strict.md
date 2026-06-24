# Duplicate Cluster Summary (strict)

Clusters are duplicate-screening groups, not proof that every member is the same image.

| Metric | Value |
| --- | --- |
| mode | strict |
| dhash_threshold | 2 |
| row_count | 14094 |
| dhash_coverage_rows | 5791 |
| file_sha256_coverage_rows | 5791 |
| cluster_count | 13673 |
| non_singleton_cluster_count | 216 |
| max_cluster_size | 100 |
| edge_count | 1380 |
| exact_file_sha_edges | 70 |
| dhash_edges | 1310 |

## Edge Category Counts

| Category | Count |
| --- | --- |
| dhash_false_positive_candidate | 673 |
| exact_file_duplicate | 70 |
| likely_near_duplicate | 637 |

## Cluster Size Distribution

| Cluster size | Cluster count |
| --- | --- |
| 1 | 13457 |
| 2 | 176 |
| 3 | 20 |
| 4 | 6 |
| 5 | 3 |
| 6 | 2 |
| 7 | 4 |
| 8 | 1 |
| 11 | 1 |
| 12 | 1 |
| 15 | 1 |
| 100 | 1 |

## Largest Clusters

| Cluster | Size | Labels | Sources |
| --- | --- | --- | --- |
| dupcluster::strict::577c15903805fd78c82c697b | 100 | {"fog": 30, "normal_day": 8, "rain": 16, "transition": 46} | {"offline_mwd": 52, "offline_weather11": 36, "offline_weather_time": 12} |
| dupcluster::strict::f07b2c0f14e746ea92eafd69 | 15 | {"fog": 5, "normal_day": 1, "rain": 2, "transition": 7} | {"offline_mwd": 8, "offline_weather11": 5, "offline_weather_time": 2} |
| dupcluster::strict::b99d97d5846f8c6663422cb9 | 12 | {"fog": 12} | {"offline_weather11": 12} |
| dupcluster::strict::d3643bc78914f95610fd9992 | 11 | {"fog": 5, "normal_day": 1, "transition": 5} | {"offline_mwd": 6, "offline_weather11": 5} |
| dupcluster::strict::6c90a7e88d853ae95eea06b4 | 8 | {"transition": 8} | {"offline_weather_time": 8} |
| dupcluster::strict::8205cdaf5866cd0ef5321db3 | 7 | {"fog": 1, "normal_day": 2, "transition": 4} | {"offline_mwd": 6, "offline_weather11": 1} |
| dupcluster::strict::e2e31c73c0f912b6efeac4a5 | 7 | {"fog": 1, "normal_day": 1, "transition": 5} | {"offline_mwd": 6, "offline_weather11": 1} |
| dupcluster::strict::9e8f1157550236d538a53b65 | 7 | {"rain": 7} | {"offline_weather_time": 7} |
| dupcluster::strict::0c7214065a12aaacfc259340 | 7 | {"normal_day": 7} | {"offline_weather_time": 7} |
| dupcluster::strict::0a96525f8a28d60defea4c9b | 6 | {"normal_day": 6} | {"offline_weather_time": 6} |
| dupcluster::strict::d0672804b78d16e74ef22491 | 6 | {"normal_day": 6} | {"offline_weather_time": 6} |
| dupcluster::strict::d5d12dca8e448d7985f99167 | 5 | {"fog": 4, "rain": 1} | {"offline_weather11": 4, "offline_weather_time": 1} |
| dupcluster::strict::6f99c4544fcfa1e4fe7dbdcc | 5 | {"normal_day": 5} | {"offline_weather_time": 5} |
| dupcluster::strict::bd42be6c663c74ff9196918a | 5 | {"normal_day": 4, "transition": 1} | {"offline_mwd": 5} |
| dupcluster::strict::b9123688faf1c4bd330aadd5 | 4 | {"normal_day": 4} | {"offline_weather_time": 4} |
| dupcluster::strict::35fa96cf39ec51103496f834 | 4 | {"normal_day": 4} | {"offline_mwd": 4} |
| dupcluster::strict::bc4e7a865d01883b957778e3 | 4 | {"fog": 4} | {"offline_weather11": 4} |
| dupcluster::strict::f81aa096945ce590105c166f | 4 | {"fog": 4} | {"offline_weather11": 4} |
| dupcluster::strict::855c9141eacfb0187817a16b | 4 | {"rain": 4} | {"offline_weather_time": 4} |
| dupcluster::strict::6aea9a623da1ad7fda7d0044 | 4 | {"rain": 4} | {"offline_weather_time": 4} |
