"""Shared helpers for the model wrappers.

All wrappers take a table from build_series.py and return a frame in the
submission schema: marketplace_id, partner_id, page_id, horizon_week (1..N),
horizon_date (Sunday), forecast.
"""

import pandas as pd

from ..core_target import PRODUCT_COLS as KEYS

ID_SEP = "|"


def pack_id(df: pd.DataFrame) -> pd.Series:
    return (
        df["marketplace_id"].astype(str)
        + ID_SEP
        + df["partner_id"].astype(str)
        + ID_SEP
        + df["page_id"].astype(str)
    )


def unpack_id(s: pd.Series) -> pd.DataFrame:
    parts = s.str.split(ID_SEP, expand=True)
    return pd.DataFrame(
        {
            "marketplace_id": parts[0].astype(int),
            "partner_id": parts[1].astype(int),
            "page_id": parts[2],
        },
        index=s.index,
    )


def to_submission(df: pd.DataFrame, valid_date: str) -> pd.DataFrame:
    """Finalize a frame with KEYS + horizon_date + forecast: daily forecasts are
    summed into Sunday weeks, horizon_week is derived, negatives clipped to 0."""
    valid = pd.Timestamp(valid_date)
    return (
        df.assign(
            horizon_date=lambda d: d["horizon_date"]
            - pd.to_timedelta((d["horizon_date"].dt.dayofweek + 1) % 7, unit="D")
        )
        .groupby(KEYS + ["horizon_date"], as_index=False)["forecast"].sum()
        .assign(
            horizon_week=lambda d: ((d["horizon_date"] - valid).dt.days // 7 + 1).astype(int),
            forecast=lambda d: d["forecast"].clip(lower=0),
        )[KEYS + ["horizon_week", "horizon_date", "forecast"]]
    )
