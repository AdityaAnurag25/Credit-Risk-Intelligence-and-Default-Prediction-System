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
```

(`make install` runs the first step. The second is a one-time local git filter that strips notebook outputs
before they're committed — run it once per clone.)

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

Model performance: TBD (Phase 2 leakage audit in progress).

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
