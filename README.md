# QM640 late-delivery capstone

Prepared locally for Pankaj Nagpal. 

This repository contains the original source audit and the executed October 27 interim analysis for Pankaj Nagpal, Group 7.

## Interim report evaluator access

- [Analysis instructions and findings](interim_2026_10_27/README.md)
- [Cleaned order-level feature data](interim_2026_10_27/processed_orders.csv)
- [Executed model and inference outputs](interim_2026_10_27/enhanced_results.json)
- [Training-only sample-size pilot](interim_2026_10_27/pilot_planning_results.json)

The interim analysis includes logistic regression, random forest, gradient boosting, adjusted associations, grouped logistic explanations, and robustness checks. These are development/validation findings. The reserved final test set has not been scored.

## Source and license

Constante, F., Silva, F., and Pereira, A. (2019). DataCo SMART SUPPLY CHAIN FOR BIG DATA ANALYSIS,
Version 5. Mendeley Data. https://doi.org/10.17632/8gx2fvg2k6.5

Source: https://data.mendeley.com/datasets/8gx2fvg2k6/5
License: CC BY 4.0. https://creativecommons.org/licenses/by/4.0/

The publisher raw CSV is deliberately not redistributed here because it contains identifying
and security-sensitive columns. The original data remain openly accessible at the publisher.
No row-level identifying values are included in this package.

## Reproduce the profiling

1. Download DataCoSupplyChainDataset.csv and DescriptionDataCoSupplyChain.csv from Version 5.
2. Keep the raw dataset private and outside any public repository.
3. Install the profiling dependency: `python -m pip install -r requirements.txt`.
4. Run `python scripts/profile_data.py --csv /path/to/DataCoSupplyChainDataset.csv --out outputs/reproduced`.
5. Compare source_profile.json and raw_missingness.csv with the supplied outputs.

Verified source SHA-256:
fa6d022ed437155e1a2f0378710602848703c8a7f203f7ff5d77805bf8480aa6

Expected: 180,519 source rows, 53 fields, 65,752 distinct orders.
The raw non-null audit does not establish range validity or pre-dispatch feature availability.

## Remaining final-report work

- Training-only tuning and calibration, operational threshold policy, customer-separated and functional-form sensitivity checks.
- Freeze the selected pipeline before reserved-test evaluation.
- Evaluate final probability quality and segment behavior; develop bounded managerial recommendations.
