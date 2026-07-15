# Runbook

Copy-pasteable commands to reproduce this project locally, end to end: install, train, serve, score.

## 1. Install

```bash
uv sync
uv run nbstripout --install
uv run pre-commit install
```

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/). Download the LendingClub CSV from Kaggle (see
the main README's "Dataset" section) and place it at `data/raw/loan.csv` before training.

## 2. Train

```bash
make train
```

Trains XGBoost (the default), evaluates on a chronological holdout, calibrates on the validation vintage,
computes SHAP explanations, and registers the model in the local MLflow registry under the "champion"
alias. Takes several minutes on the full ~2.2M-row dataset. Expected tail of the log output:

```
... INFO Metrics: {'roc_auc': 0.71..., 'ks_statistic': 0.31..., 'gini_coefficient': 0.43..., ...}
... INFO Brier score — raw: 0.22... | isotonic-calibrated: 0.14...
... INFO Registered credit_risk_pd_model v1 as 'champion'
```

(Exact metrics vary slightly run to run — see the main README's "Model Performance" section for the
canonical reported numbers. `v1` is what a first-ever registration looks like; re-running `make train`
registers `v2`, `v3`, etc.)

Train the other two models the same way if you want to compare:

```bash
make train MODEL=logistic
make train MODEL=random_forest
```

## 3. Serve

```bash
make serve
```

Starts the API on `http://localhost:8000`. Loads the "champion" model at startup, so `make train` must
have registered one first.

## 4. Score a request

```bash
curl http://localhost:8000/health
```

Expected:

```json
{"status": "ok", "model_version": "1"}
```

```bash
curl -X POST http://localhost:8000/score \
  -H "Content-Type: application/json" \
  -d '{
    "loan_amnt": 15000, "term": "36 months", "int_rate": 13.5, "installment": 450.0,
    "grade": "B", "sub_grade": "B3", "emp_length": "5 years", "home_ownership": "RENT",
    "annual_inc": 65000, "verification_status": "Verified", "purpose": "debt_consolidation",
    "dti": 18.5, "delinq_2yrs": 0, "earliest_cr_line": "Jan-2010", "inq_last_6mths": 1,
    "open_acc": 8, "pub_rec": 0, "revol_bal": 12000, "revol_util": 45.0, "total_acc": 20,
    "initial_list_status": "w", "application_type": "Individual"
  }'
```

Expected shape (exact numbers depend on your trained model, but should be in this neighborhood for a
grade-B, low-DTI applicant):

```json
{
  "pd": 0.145,
  "risk_band": "Very Low",
  "decision": "Approve",
  "top_reasons": [
    {"feature": "sub_grade", "value": "B3", "direction": "decreases_risk", "magnitude": 0.262},
    {"feature": "term", "value": "36 months", "direction": "decreases_risk", "magnitude": 0.154},
    {"feature": "grade", "value": "B", "direction": "decreases_risk", "magnitude": 0.104},
    {"feature": "home_ownership", "value": "RENT", "direction": "increases_risk", "magnitude": 0.074},
    {"feature": "total_bc_limit", "value": 15000.0, "direction": "decreases_risk", "magnitude": 0.035}
  ]
}
```

A malformed request (missing/invalid fields) returns `422` with per-field validation errors:

```bash
curl -X POST http://localhost:8000/score -H "Content-Type: application/json" -d '{"loan_amnt": -5}'
```

## 5. Everything else

```bash
make evaluate MODEL_PATH=outputs/models/xgboost_<timestamp>.pkl   # re-evaluate a saved model
make explain MODEL_PATH=outputs/models/xgboost_<timestamp>.pkl    # SHAP summary/bar plots
make monitor                                                      # Evidently drift report
make mlflow-ui                                                    # browse runs at http://localhost:5000
make test                                                         # full pytest suite
make lint                                                         # ruff check
```

See the main [README](../README.md) for what each of these does, and `Makefile` for the full target list.

## 6. Recording the API demo

For the FastAPI portfolio artifact — a short screen recording, not staged screenshots. Target length:
60–90 seconds.

**Tool:** [peek](https://github.com/phw/peek) if you're on Linux and want a GIF — records a window region,
exports straight to `.gif`, no editing step. For a text-only cast that embeds directly in a README instead of
a video/GIF file, use [asciinema](https://asciinema.org/) (`asciinema rec demo.cast`) — either embed the
hosted player, or run it through [svg-term-cli](https://github.com/marionebl/svg-term-cli) for a static SVG
with no JS dependency.

Before hitting record, start the MLflow UI in a separate terminal so it's already loaded in a browser tab —
`compose-up` doesn't start it:

```bash
make mlflow-ui
```

Then record, in order:

```bash
make compose-up
curl http://localhost:8000/health
curl -X POST http://localhost:8000/score \
  -H "Content-Type: application/json" \
  -d '{
    "loan_amnt": 15000, "term": "36 months", "int_rate": 13.5, "installment": 450.0,
    "grade": "B", "sub_grade": "B3", "emp_length": "5 years", "home_ownership": "RENT",
    "annual_inc": 65000, "verification_status": "Verified", "purpose": "debt_consolidation",
    "dti": 18.5, "delinq_2yrs": 0, "earliest_cr_line": "Jan-2010", "inq_last_6mths": 1,
    "open_acc": 8, "pub_rec": 0, "revol_bal": 12000, "revol_util": 45.0, "total_acc": 20,
    "initial_list_status": "w", "application_type": "Individual"
  }'
```

- Open `http://localhost:8000/docs` — the Swagger UI generated from `LoanApplication` / `ScoreResponse`
  (`src/credit_risk/api/schemas.py`). Worth expanding `POST /score` on camera to show the schema.
- Open `http://localhost:5000` — the MLflow UI tab from before recording. Click into the champion run to
  show its logged params/metrics.

Stop recording, then clean up off-camera:

```bash
make compose-down
```
