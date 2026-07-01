"""Partner x marketplace aggregate series for hierarchical reconciliation (weekly +
daily, the latter carrying partner_units/page_views covariates)."""
from pathlib import Path

from .build_series import EVENT_COLS, build_series
from .core_target import PRODUCT_COLS

AGG_PAGE_ID = "_AGG_"


def build_agg_series(core_path: Path, valid_date: str):
    wk = build_series(core_path, valid_date, "W")
    agg = {"target": ("target", lambda s: s.sum(min_count=1))} | {c: (c, "max") for c in EVENT_COLS}
    return (
        wk.groupby(["marketplace_id", "partner_id", "week_start"], as_index=False).agg(**agg)
        .assign(page_id=AGG_PAGE_ID, promo_pct_off=0.0)
        [PRODUCT_COLS + ["week_start", "target", "promo_pct_off"] + EVENT_COLS]
    )


def build_agg_series_daily(daily_series):
    """DAILY partner x marketplace aggregate from a per-product daily series (freq=D,
    with partner_units + page_views), carrying the past covariates."""
    agg = {"target": ("target", lambda s: s.sum(min_count=1))}
    if "partner_units" in daily_series.columns:
        agg["partner_units"] = ("partner_units", "first")
    if "page_views" in daily_series.columns:
        agg["page_views"] = ("page_views", "sum")
    agg |= {c: (c, "max") for c in EVENT_COLS if c in daily_series.columns}
    return (
        daily_series.groupby(["marketplace_id", "partner_id", "date"], as_index=False).agg(**agg)
        .assign(page_id=AGG_PAGE_ID, promo_pct_off=0.0)
    )
