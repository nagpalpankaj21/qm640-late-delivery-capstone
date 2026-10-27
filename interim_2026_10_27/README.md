# QM640 interim analysis - October 27, 2026

Pankaj Nagpal, Group 7. This folder contains executed interim analyses, a non-identifying processed feature dataset, and aggregate results. Final reserved-test performance and production benefits are not claimed.

## Evaluator access

- [Processed feature data](processed_orders.csv): 65,752 order records, including outcome and partition labels; no original order/customer identifiers, addresses, passwords, or placement timestamps.
- [Primary analysis results](enhanced_results.json): model performance, uncertainty, associations, explanations and sensitivity checks.
- [Source audit](analysis_results.json) and [raw-field dictionary](raw_data_dictionary.csv).
- [Training-only sample-size pilot](pilot_planning_results.json): 3-fold out-of-fold variability estimates using 39,450 training orders only.

Raw data: [DataCo Version 5](https://data.mendeley.com/datasets/8gx2fvg2k6/5). Download DataCoSupplyChainDataset.csv and DescriptionDataCoSupplyChain.csv there. Raw identifying records are not redistributed.

## Reproduce

Use Python 3.12.14 and install `pip install -r requirements.txt` in a clean environment.

1. `python baseline_audit.py --data /path/to/DataCoSupplyChainDataset.csv`
2. `python enhance_analysis.py /path/to/DataCoSupplyChainDataset.csv`
3. `python pilot_planning.py`

Outputs are written beside the scripts. The initial audit verifies the source SHA-256 before analysis. The private order/customer grouping keys and timestamps needed for clustered and chronological checks are regenerated from the raw source locally.

Expected split: training 39,450; validation 13,151; reserved test 13,151. Seeds: partition 640; order bootstrap 642; customer bootstrap 643; training-only pilot 644. No reserved-test prediction metrics are generated. The pilot estimates variability; it does not tune hyperparameters or establish operational effect sizes.

## Interpretation and attribution

Expanded logistic validation AUC = 0.73964 and Brier = 0.19509. Random forest and gradient boosting have similar performance, with no clear superiority in the adjusted paired probability-loss intervals. Shipping mode supplies most predictive signal. Within-mode discrimination is weak in several groups. Explanations are additive logistic log-odds contributions, not causal effects or explanations of the tree models.

Constante, F., Silva, F., & Pereira, A. (2019). DataCo SMART SUPPLY CHAIN FOR BIG DATA ANALYSIS (Version 5). Mendeley Data. https://doi.org/10.17632/8gx2fvg2k6.5

CC BY 4.0. Transformations: source auditing, order aggregation, removal of identifying fields, feature engineering, fixed partitions, statistical models, and aggregate outputs.
