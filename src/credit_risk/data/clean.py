import numpy as np
import pandas as pd

TARGET_STATUSES: tuple[str, ...] = ("Fully Paid", "Charged Off")


def clean_and_label(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    text_cols = df.select_dtypes(include=["object"]).columns
    for col in text_cols:
        df[col] = df[col].astype(str).str.strip()

    df = df[df["loan_status"].isin(TARGET_STATUSES)]

    df["target_default"] = np.where(df["loan_status"] == "Charged Off", 1, 0)

    return df
