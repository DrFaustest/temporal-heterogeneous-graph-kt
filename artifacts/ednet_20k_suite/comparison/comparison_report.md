# Comparison Report

Best run by AUC: **ednet20k_thgkt_no_prereq**

| Rank | Run | Model | Relation Mode | AUC | Accuracy | F1 | BCE Loss |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| 1 | ednet20k_thgkt_no_prereq | thgkt | transition | 0.7635 | 0.7311 | 0.8174 | 0.5325 |
| 2 | ednet20k_thgkt_cooccurrence_graph | thgkt | cooccurrence | 0.7619 | 0.7301 | 0.8169 | 0.5345 |
| 3 | ednet20k_thgkt_full | thgkt | transition | 0.7613 | 0.7299 | 0.8160 | 0.5348 |
| 4 | ednet20k_thgkt_no_temporal | thgkt | transition | 0.7539 | 0.7249 | 0.8101 | 0.5408 |
| 5 | ednet20k_dkt_baseline | dkt_baseline | transition | 0.6423 | 0.6914 | 0.8054 | 0.6039 |
| 6 | ednet20k_graph_only | graph_only | transition | 0.5858 | 0.6709 | 0.8017 | 0.6241 |
