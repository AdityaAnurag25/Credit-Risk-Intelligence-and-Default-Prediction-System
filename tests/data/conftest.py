import pandas as pd
import pytest


@pytest.fixture
def raw_loans_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "loan_status": [
                "Fully Paid",
                "Charged Off",
                "Current",
                "Late (31-120 days)",
                "In Grace Period",
                "Fully Paid",
                "Charged Off",
                "Late (16-30 days)",
                "Default",
                "Fully Paid",
                "Charged Off",
                "Current",
                "Fully Paid",
                "Charged Off",
                "In Grace Period",
            ],
            "loan_amnt": [
                1000,
                2000,
                1500,
                3000,
                2500,
                1000,
                5000,
                1200,
                900,
                1800,
                2200,
                3100,
                1600,
                2700,
                2100,
            ],
            "grade": [
                " A ",
                "B",
                "C ",
                " D",
                "E",
                "A",
                "B",
                "C",
                "D",
                "E",
                "A",
                "B",
                "C",
                "D",
                "E",
            ],
        }
    )
