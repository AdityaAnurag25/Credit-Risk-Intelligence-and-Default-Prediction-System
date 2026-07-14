import logging

import pandas as pd

logger = logging.getLogger(__name__)

# Post-origination / hardship / servicing-time columns. A borrower's post-loan
# outcome (charge-off, hardship enrollment, debt settlement) directly
# determines these values, so they leak the target into the feature matrix.
#
# Verified against the actual data/raw/loan.csv header. `last_fico_range_high`
# and `last_fico_range_low` don't exist in this particular Kaggle export (it
# omits FICO entirely) but are kept here for forward-compatibility with
# LendingClub exports that do include them — drop_leaky_columns() ignores
# columns that aren't present.
LEAKY_COLUMNS: tuple[str, ...] = (
    "collection_recovery_fee",
    "debt_settlement_flag",
    "debt_settlement_flag_date",
    "funded_amnt",
    "funded_amnt_inv",
    "hardship_amount",
    "hardship_dpd",
    "hardship_end_date",
    "hardship_flag",
    "hardship_last_payment_amount",
    "hardship_length",
    "hardship_loan_status",
    "hardship_payoff_balance_amount",
    "hardship_reason",
    "hardship_start_date",
    "hardship_status",
    "hardship_type",
    "last_credit_pull_d",
    "last_fico_range_high",
    "last_fico_range_low",
    "last_pymnt_amnt",
    "last_pymnt_d",
    "next_pymnt_d",
    "out_prncp",
    "out_prncp_inv",
    "pymnt_plan",
    "recoveries",
    "settlement_amount",
    "settlement_date",
    "settlement_percentage",
    "settlement_status",
    "settlement_term",
    "total_pymnt",
    "total_pymnt_inv",
    "total_rec_int",
    "total_rec_late_fee",
    "total_rec_prncp",
)


def drop_leaky_columns(df: pd.DataFrame) -> pd.DataFrame:
    present = [c for c in LEAKY_COLUMNS if c in df.columns]
    if present:
        logger.info("Dropping %d leaky column(s): %s", len(present), present)
    return df.drop(columns=present)
