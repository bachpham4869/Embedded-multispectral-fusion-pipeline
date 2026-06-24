# Feature-Vector Overlap Details

Status: warning-level leakage evidence. These rows have identical `optical_12_baseline` feature vectors across train/test, but image-level duplicate status cannot be confirmed without original image path/hash metadata.

Overlap pairs: 22

CSV artifact: `artifacts/ml/leakage/feature_vector_overlaps.csv`

| Train idx | Test idx | Train label | Test label | Train source | Test source | Train conf | Test conf | NIR | Thermal | Feature hash | Match | Missing metadata |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 7427 | 114 | fog | fog | offline_weather11 | offline_weather11 | 0.6 | 0.6 | rgb | none | 6dbf61bcc781 | exact_feature_vector | original image path/hash metadata; session_id metadata |
| 10958 | 206 | rain | rain | offline_mwd | offline_mwd | 0.9 | 0.9 | rgb | none | eab19213143c | exact_feature_vector | original image path/hash metadata; session_id metadata |
| 3170 | 331 | fog | fog | offline_weather11 | offline_weather11 | 0.6 | 0.6 | rgb | none | 044c22eca660 | exact_feature_vector | original image path/hash metadata; session_id metadata |
| 5607 | 456 | fog | fog | offline_weather11 | offline_weather11 | 0.85 | 0.85 | rgb | none | f0901c26fad1 | exact_feature_vector | original image path/hash metadata; session_id metadata |
| 4249 | 606 | fog | fog | offline_weather11 | offline_weather11 | 0.85 | 0.85 | rgb | none | fd1151151df3 | exact_feature_vector | original image path/hash metadata; session_id metadata |
| 4086 | 820 | fog | fog | offline_weather11 | offline_weather11 | 0.85 | 0.85 | rgb | none | 2add1f249a6a | exact_feature_vector | original image path/hash metadata; session_id metadata |
| 11820 | 930 | rain | rain | offline_mwd | offline_mwd | 0.9 | 0.9 | rgb | none | 368efa6d96f9 | exact_feature_vector | original image path/hash metadata; session_id metadata |
| 8980 | 1019 | fog | fog | offline_weather11 | offline_weather11 | 0.85 | 0.85 | rgb | none | cbd54c881287 | exact_feature_vector | original image path/hash metadata; session_id metadata |
| 5134 | 1035 | rain | rain | offline_mwd | offline_mwd | 0.9 | 0.9 | rgb | none | 226aca795c0e | exact_feature_vector | original image path/hash metadata; session_id metadata |
| 5412 | 1173 | fog | fog | offline_weather11 | offline_weather11 | 0.85 | 0.85 | rgb | none | 7c827cfde2fe | exact_feature_vector | original image path/hash metadata; session_id metadata |
| 10577 | 1222 | rain | rain | offline_weather11 | offline_weather11 | 0.9 | 0.9 | rgb | none | d82338c89876 | exact_feature_vector | original image path/hash metadata; session_id metadata |
| 1368 | 1230 | fog | fog | offline_weather11 | offline_weather11 | 0.85 | 0.85 | rgb | none | ef82b80fba91 | exact_feature_vector | original image path/hash metadata; session_id metadata |
| 5773 | 1230 | fog | fog | offline_weather11 | offline_weather11 | 0.85 | 0.85 | rgb | none | ef82b80fba91 | exact_feature_vector | original image path/hash metadata; session_id metadata |
| 1363 | 1380 | transition | transition | offline_mwd | offline_mwd | 0.8 | 0.8 | rgb | none | 8efbf9198ef6 | exact_feature_vector | original image path/hash metadata; session_id metadata |
| 3653 | 1423 | normal_day | normal_day | offline_mwd | offline_mwd | 0.85 | 0.85 | rgb | none | 517beeee70d3 | exact_feature_vector | original image path/hash metadata; session_id metadata |
| 10486 | 1475 | rain | rain | offline_mwd | offline_mwd | 0.9 | 0.9 | rgb | none | 6d93871ad233 | exact_feature_vector | original image path/hash metadata; session_id metadata |
| 2296 | 1527 | fog | fog | offline_weather11 | offline_weather11 | 0.85 | 0.85 | rgb | none | 47b71c62880e | exact_feature_vector | original image path/hash metadata; session_id metadata |
| 8322 | 1678 | transition | transition | offline_mwd | offline_mwd | 0.8 | 0.8 | rgb | none | d725c81d1e5c | exact_feature_vector | original image path/hash metadata; session_id metadata |
| 706 | 1724 | fog | fog | offline_weather11 | offline_weather11 | 0.6 | 0.6 | rgb | none | e7a284103ed8 | exact_feature_vector | original image path/hash metadata; session_id metadata |
| 3634 | 1763 | rain | rain | offline_mwd | offline_mwd | 0.9 | 0.9 | rgb | none | 11d237832a09 | exact_feature_vector | original image path/hash metadata; session_id metadata |
| 11531 | 1789 | rain | rain | offline_weather11 | offline_weather11 | 0.9 | 0.9 | rgb | none | 711115b096f0 | exact_feature_vector | original image path/hash metadata; session_id metadata |
| 4173 | 1894 | transition | transition | offline_mwd | offline_mwd | 0.8 | 0.8 | rgb | none | a9985123de32 | exact_feature_vector | original image path/hash metadata; session_id metadata |

Feature values and per-feature diffs are stored in the CSV as JSON fields.