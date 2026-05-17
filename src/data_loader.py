from pathlib import Path
import pandas as pd
from sqlalchemy import create_engine

print("Script started", flush=True)

BASE_DIR = Path(__file__).resolve().parents[1]
csv_path = BASE_DIR / "data" / "processed" / "feature_engineered_loan_data.csv"

print(f"CSV path: {csv_path}", flush=True)

username = "postgres"
password = "postgres123"
host = "localhost"
port = "5432"
database = "credit_risk_db"

engine = create_engine(f"postgresql://{username}:{password}@{host}:{port}/{database}")
print("Connected to PostgreSQL", flush=True)

df = pd.read_csv(csv_path, low_memory=False)
print(f"Dataset loaded: {df.shape}", flush=True)

sql_df = df[
    [
        "loan_amnt", "funded_amnt", "term", "int_rate", "installment",
        "grade", "sub_grade", "emp_length", "home_ownership", "annual_inc",
        "verification_status", "purpose", "dti", "revol_util", "delinq_2yrs",
        "inq_last_6mths", "open_acc", "pub_rec", "total_acc", "target_default",
        "income_to_loan_ratio", "installment_to_income", "credit_exposure_score"
    ]
].copy()

print("Starting upload...", flush=True)

sql_df.to_sql(
    "fact_loans",
    engine,
    if_exists="replace",
    index=False,
    method="multi",
    chunksize=10000
)

print("Upload completed", flush=True)