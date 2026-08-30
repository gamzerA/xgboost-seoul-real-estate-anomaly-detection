# XGBoost-Based Seoul Real Estate Price Prediction and Anomalous Transaction Screening

This project uses Seoul apartment transaction records to estimate expected sale
prices and identify transactions with unusually large gaps between actual and
predicted prices for further review. The original study was presented orally at
the 59th Spring Conference of the Korea Institute of Information and
Communication Engineering on May 22, 2026.

> An "anomalous transaction candidate" in this project is an observation that
> deviates substantially from the learned price pattern. It is not a
> determination of fraud or illegality, nor an assessment of any party involved
> in a transaction.

![Actual versus predicted transaction prices](results/figures/actual_vs_predicted.png)

## Research Question

Can Seoul apartment prices be estimated from multiple factors—including floor
area, location, floor level, construction year, and transaction date—and can
the gap between predicted and actual prices be used to prioritize transactions
for additional review?

## Data and Baseline Method

- Data: Seoul apartment transaction records from the Ministry of Land,
  Infrastructure and Transport, 2022–2025
- Transactions after preprocessing: 189,864
- Features: exclusive-use area, floor, construction year, building age,
  contract year, contract month, district, statutory neighborhood
  (`beopjeong-dong`), and
  apartment complex
- Model: `XGBRegressor`
- Split: 80% train / 20% test, `random_state=42`
- Validation: shuffled 5-fold cross-validation
- Candidate selection: top 922 transactions by absolute residual

### Baseline XGBoost Configuration

| Parameter | Value |
|---|---:|
| `n_estimators` | 300 |
| `learning_rate` | 0.05 |
| `max_depth` | 6 |
| `subsample` | 0.8 |
| `colsample_bytree` | 0.8 |

## Stored Baseline Results

The metrics preserved in the repository are:

| Metric | Value |
|---|---:|
| Test R² | 0.9258 |
| Test MAE | 14,742.59 (KRW 10,000 units) |
| Test RMSE | 24,945.73 (KRW 10,000 units) |
| Mean 5-fold CV R² | 0.9301 |

The conference slides report a mean 5-fold CV R² of `0.9172` from the original
run. The reference CSV in this repository comes from a later run and therefore
contains a different value. Locking the exact data version and execution
environment remains necessary for complete reproduction of the conference
result.

![Feature importance](results/figures/feature_importance.png)

Exclusive-use area and district had the highest importance, followed by
construction year, statutory neighborhood, and building age.

## Repository Structure

```text
.
├── data/
│   ├── README.md
│   └── raw/                       # Downloaded CSV files; excluded from Git
├── docs/
│   ├── advanced_methodology.md
│   └── presentation.pptx
├── notebooks/
│   └── xgboost_baseline.ipynb
├── results/
│   ├── figures/
│   └── reference/
├── src/
│   ├── train_xgboost.py          # Original conference baseline
│   └── train_robust_pipeline.py  # Advanced model-selection pipeline
├── tests/
│   └── test_robust_pipeline.py
├── requirements.txt
└── README.md
```

## Installation

Python 3.10 or later is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Follow the [data instructions](data/README.md) and place the CSV files in
`data/raw/` before running either pipeline.

## Run the Conference Baseline

```bash
python src/train_xgboost.py \
  --data-dir data/raw \
  --output-dir results/generated \
  --top-n 922
```

Row-level predictions and review candidates are written to
`results/generated/` and are excluded from Git.

## Advanced Pipeline: Model Selection and Robustness Validation

The conference baseline remains unchanged. The follow-up workflow is provided
separately in [`src/train_robust_pipeline.py`](src/train_robust_pipeline.py).

- Candidate models: median baseline, Ridge, Random Forest, Histogram Gradient
  Boosting, and XGBoost
- Model selection: expanding-window temporal validation using only years before
  the final holdout year
- Final evaluation: a one-time out-of-time evaluation on the most recent year
- Leakage prevention: cross-fitted Target Encoding inside each training fold,
  with explicit handling of previously unseen categories
- Robustness analysis: 95% bootstrap confidence intervals; performance by
  district, area, and price segment; multi-seed metric sensitivity; and
  candidate-set stability
- Candidate scoring: out-of-time residuals and year-normalized robust log-price
  residual scores instead of predictions on training rows
- Reproducibility: random seeds, package versions, and SHA-256 hashes of input
  CSV files recorded for every run

Run the advanced pipeline on the complete dataset:

```bash
python src/train_robust_pipeline.py \
  --data-dir data/raw \
  --output-dir results/advanced/generated \
  --top-n 922
```

For a quick structural check during development, use a stratified sample from
each year. Metrics produced with this option must not be reported as final
research results.

```bash
python src/train_robust_pipeline.py \
  --data-dir data/raw \
  --output-dir results/advanced/smoke \
  --sample-per-year 1000 \
  --bootstrap-iterations 30
```

See [`docs/advanced_methodology.md`](docs/advanced_methodology.md) for the full
validation design and output descriptions.

## Limitations and Future Work

- External variables such as transport accessibility, school districts,
  interest rates, redevelopment plans, and remodeling status are not included.
- Label Encoding in the conference baseline can impose an arbitrary order on
  categories. The advanced pipeline addresses this with fold-internal Target
  Encoding.
- The robust log-residual score prioritizes unusual observations but does not
  explain why a transaction is unusual.
- The conference baseline scores the full dataset with a single fitted model.
  The advanced pipeline addresses this optimism by using out-of-time
  predictions.
- Temporal and subgroup evaluation are implemented, but separate district-level
  model comparisons remain future work.
- No authoritative labels for fraudulent or illegal transactions are available,
  so detection precision and recall cannot yet be estimated. External variables
  and expert-reviewed samples are needed for further validation.

## Presentation

- [Conference presentation slides](docs/presentation.pptx)

## Sources

- [MOLIT Real Estate Transaction Price Disclosure System](https://rt.molit.go.kr/)
- [scikit-learn documentation](https://scikit-learn.org/)
- [XGBoost documentation](https://xgboost.readthedocs.io/)
