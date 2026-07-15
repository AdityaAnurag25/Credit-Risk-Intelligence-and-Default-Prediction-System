# Credit Risk Intelligence & Default Prediction System

An end-to-end credit risk pipeline built on LendingClub's historical consumer loan data. Raw loan-level
records are cleaned, feature-engineered, and used to train Logistic Regression, Random Forest, and XGBoost
classifiers that estimate probability of default. The reusable pipeline logic lives in `src/credit_risk/`;
the notebooks in `notebooks/` are thin, narrated drivers over that same code, not a separate implementation.

## Problem Framing

Lenders need to estimate a borrower's probability of default using only information available at
application time. This project frames that as a binary classification problem — `Fully Paid` vs.
`Charged Off` — and builds a pipeline to train, evaluate, and operationalize models against it.

## Dataset

[LendingClub Loan Data](https://www.kaggle.com/datasets/adarshsng/lending-club-loan-data-csv) (Kaggle).
Download it and place the CSV at `data/raw/loan.csv`. Raw and processed data are gitignored and not
included in this repo.

## Setup

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run nbstripout --install
uv run pre-commit install
```

(`make install` runs the first step. The second is a one-time local git filter that strips notebook outputs
before they're committed — run it once per clone. The third installs the pre-commit hooks — ruff, end-of-file
and trailing-whitespace fixers, a large-file check, and nbstripout — so they run automatically on `git commit`.)

Run `make test` to verify the install.

## Usage

Train a model (defaults to XGBoost):

```bash
make train
make train MODEL=logistic   # or random_forest
```

Trained models are saved to `outputs/models/`. Evaluate a saved model against a freshly rebuilt holdout
split:

```bash
make evaluate MODEL_PATH=outputs/models/xgboost_<timestamp>.pkl
```

## Model Performance

XGBoost, evaluated on a chronological holdout of `issue_d` (train: 2007-06 to 2016-03, validation: 2016-03
to 2017-01, test: 2017-01 to 2018-12), after removing all identified leakage columns:

| Metric | Value |
|---|---|
| Test ROC-AUC | 0.718 |
| Test KS statistic | 0.313 |
| Test Gini coefficient | 0.436 |
| Brier score, raw → isotonic-calibrated | 0.225 → 0.150 |

Performance is stable (~0.70–0.73 AUC) across 2017 through mid-2018 vintages, then degrades sharply in the
final months of 2018 — this tracks label immaturity near the data cutoff (defaults take 12–18+ months to
materialize, so the most recent vintages' "Charged Off" labels are a biased, fast-defaulter-only sample),
not a real drop in model quality. See `outputs/figures/calibration_xgboost.png` for the reliability diagram
and `outputs/figures/shap_summary.png` for global feature attribution.

### Modelling Notes

- **Leakage handling:** drops 35 post-origination/servicing columns (`total_pymnt`, `recoveries`,
  `out_prncp`, the `hardship_*` and `settlement_*` families, `funded_amnt`, etc. — see
  `src/credit_risk/data/leakage.py`) before feature engineering. This is what took the reported AUC from a
  leakage-inflated ~0.82 down to the honest ~0.72 above.
- **Time-based split:** train/val/test are chronological slices of `issue_d` (70/15/15), not a random split
  — a random split lets the model see future vintages during training, which doesn't match deployment.
- **Class imbalance:** XGBoost uses `scale_pos_weight` computed from the train-set class ratio; Logistic
  Regression uses `class_weight="balanced"`. No SMOTE.
- **Calibration:** isotonic regression fit on the validation vintage (`CalibratedClassifierCV` +
  `FrozenEstimator`), evaluated by Brier score on the test set — the source of the 0.225 → 0.150 improvement
  above.

## Project Structure

```
.
├── data/               # raw/processed loan data (gitignored; see Dataset)
├── notebooks/          # thin-driver notebooks: narrative + calls into src/credit_risk
├── outputs/            # metrics, figures, predictions, trained models (gitignored)
├── reports/            # written summaries
├── scripts/            # CLI entrypoints: train.py, evaluate.py
├── sql/                # portfolio and risk analysis queries
├── src/credit_risk/    # installable package
│   ├── data/           # loading, cleaning, audit
│   ├── features/       # feature engineering
│   ├── models/         # train, evaluate, decisioning
│   ├── validation/     # empty scaffold, no logic yet
│   └── utils/          # empty scaffold, no logic yet
├── tests/              # pytest suite, mirrors src/credit_risk
├── Makefile
├── pyproject.toml
└── uv.lock
```

## Tech Stack

Core (from `pyproject.toml`):
- pandas, numpy — data handling
- scikit-learn, xgboost — modeling
- sqlalchemy, psycopg2-binary — PostgreSQL connectivity
- pydantic, pydantic-settings — typed config (`src/credit_risk/config.py`)
- python-dotenv, joblib

Development:
- ruff — lint and format
- mypy — type checking
- pytest, pytest-cov — testing
- jupyterlab, ipykernel — notebooks
- matplotlib, seaborn — notebook visualization
- nbstripout — strips notebook outputs before commit

## License

[MIT](LICENSE) © 2026 Aditya Anurag
