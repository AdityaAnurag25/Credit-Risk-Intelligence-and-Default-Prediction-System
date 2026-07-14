from pathlib import Path

import pandas as pd


def load_raw_data(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)
