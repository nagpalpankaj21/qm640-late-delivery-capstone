# QM640 synopsis reproducibility support

Prepared locally for Pankaj Nagpal. No GitHub or GitLab repository was created.

This package contains verified source-data profiling, a complete 53-field raw dictionary,
and measured missingness. Predictive modeling, inference, and SHAP analysis are proposed
capstone work; this package does not claim those analyses have been performed.

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

## Proposed analysis sequence

- Validate pre-dispatch availability and within-order metadata consistency.
- Aggregate line items to one order, using the rules in the synopsis.
- Freeze independent training, validation, and test orders.
- Estimate RQ1 associations; tune and calibrate RQ2 models within development data.
- Compare paired test Brier losses for RQ3; obtain RQ4 explanations and uncertainty.
- Publish aggregate outputs, code, executed environment versions, and limitations.

## Optional later publication

The template's public GitHub requirement remains outstanding. If you choose to publish,
create a repository in your own account, upload this support package and later analysis code,
keep raw personal/security-sensitive data out of the repository, and provide the publisher
download instructions. Add the actual public repository URL to the synopsis before submission
if the course requires it at this stage. Do not describe this local package as a published repository.
