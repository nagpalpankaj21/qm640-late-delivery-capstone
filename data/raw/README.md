# Raw source access and local path

Source: Constante, F., Silva, F., & Pereira, A. (2019), DataCo SMART SUPPLY CHAIN FOR BIG DATA ANALYSIS, Version 5.
https://data.mendeley.com/datasets/8gx2fvg2k6/5
https://doi.org/10.17632/8gx2fvg2k6.5

Download these exact files from the Version 5 Files section:
1. DataCoSupplyChainDataset.csv (structured observations)
2. DescriptionDataCoSupplyChain.csv (field descriptions)

Read the structured CSV with Latin-1 encoding. Its expected SHA-256 is:
fa6d022ed437155e1a2f0378710602848703c8a7f203f7ff5d77805bf8480aa6

Local raw path: data/raw/DataCoSupplyChainDataset.csv and data/raw/DescriptionDataCoSupplyChain.csv, or another private folder outside this repository. Raw CSVs are ignored by Git. This README is the repository raw-data access pointer; it does not claim the identifying raw records are hosted here.

From the repository root, reproduce with:
python interim_2026_10_27/baseline_audit.py --data /private/path/DataCoSupplyChainDataset.csv
python interim_2026_10_27/enhance_analysis.py /private/path/DataCoSupplyChainDataset.csv
python interim_2026_10_27/pilot_planning.py
python interim_2026_10_27/audit_outcome_and_bootstrap.py /private/path/DataCoSupplyChainDataset.csv

Cleaned/processed data are publicly hosted at interim_2026_10_27/processed_orders.csv. A small selected-column training preview is interim_2026_10_27/processed_preview.csv.

License: CC BY 4.0. The analytical export removes identifying fields and aggregates line items to orders. Obtain the original identifying records directly from the attributed publisher; do not commit them.
