import pandas as pd


def generate_audit_table(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "column": df.columns,
            "dtype": df.dtypes.values,
            "missing_count": df.isnull().sum().values,
            "missing_percent": (df.isnull().sum().values / len(df)) * 100,
            "unique_values": [df[col].nunique() for col in df.columns],
        }
    )
