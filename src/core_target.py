"""Demand-history source: in-stock units from core_daily (`units` on non-OOS
days, before valid_date). core_daily is the only input table the final test
provides; OOS days are dropped (treated as a gap by consumers).
"""

from pathlib import Path

import pandas as pd

PRODUCT_COLS = ["marketplace_id", "partner_id", "page_id"]
CORE_DAILY_PATH = Path("data/raw/forecasting_hackathon_core_daily.parquet")


def load_core_instock(
    core_path: Path = CORE_DAILY_PATH, valid_date: str | None = None
) -> pd.DataFrame:
    """Daily in-stock units from core_daily: (product, date, units) on non-OOS
    days, optionally clamped to dates strictly before valid_date."""
    df = (pd.read_parquet(core_path, columns=PRODUCT_COLS + ["date", "units", "oos"])
          .rename(columns=str.lower)
          .astype({"marketplace_id": int, "partner_id": int, "page_id": str})
          .assign(date=lambda d: pd.to_datetime(d["date"])))
    instock = df["oos"].fillna(0) == 0  # in-stock days only (eval scores these)
    if valid_date is not None:
        instock &= df["date"] < pd.Timestamp(valid_date)
    return df.loc[instock, PRODUCT_COLS + ["date", "units"]]
