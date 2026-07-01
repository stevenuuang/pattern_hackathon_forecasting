"""Modeling table at daily or weekly grain: one row per product per period over
history + the 13-week horizon. target = in-stock units from core_daily (NaN
on/after valid_date); covariates from core_daily (promo masked beyond
valid_date+28d, known ~4wk out). Optional page-level past covariates
partner_units + page_views; uncensor (weekly) extrapolates demand over
OOS-censored days.
"""

from pathlib import Path

import pandas as pd

from .core_target import PRODUCT_COLS

EVENT_COLS = ["prime_day", "big_deals", "black_fri", "cyber_mon", "christmas", "ny_day"]
PAST_ONLY_COLS = ["avg_price_paid", "buybox_pct", "buybox_suppression_pct", "ad_spend", "oos"]
PROMO_KNOWN_DAYS = 28


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.rename(columns=str.lower)
        .astype({"marketplace_id": int, "partner_id": int, "page_id": str})
        .assign(date=lambda d: pd.to_datetime(d["date"]))
    )


def _add_partner_units(df: pd.DataFrame, ext_path: Path, valid: pd.Timestamp) -> pd.DataFrame:
    """Past-only covariate: partner x marketplace daily units, full catalog."""
    partner = (
        pd.read_parquet(ext_path, columns=["marketplace_id", "partner_id", "date", "partner_units"])
        .assign(date=lambda d: pd.to_datetime(d["date"]))
        .query("date < @valid")
        .groupby(["marketplace_id", "partner_id", "date"], as_index=False)["partner_units"].sum()
        .astype({"marketplace_id": int, "partner_id": int})
    )
    return df.merge(partner, on=["marketplace_id", "partner_id", "date"], how="left").assign(
        partner_units=lambda d: d["partner_units"].mask(d["date"] < valid, d["partner_units"].fillna(0.0))
    )


def _add_pageviews(df: pd.DataFrame, ext_path: Path, valid: pd.Timestamp) -> pd.DataFrame:
    """Past-only covariate: page-level daily glance views — demand-leading
    traffic the units history can't see (conversion varies, so it is not a
    units proxy). One row per (marketplace, page, day) in the source."""
    pageviews = (
        pd.read_parquet(ext_path, columns=["marketplace_id", "page_id", "date", "page_views"])
        .assign(date=lambda d: pd.to_datetime(d["date"]))
        .query("date < @valid")
        .astype({"marketplace_id": int, "page_id": str})
    )
    return df.merge(pageviews, on=["marketplace_id", "page_id", "date"], how="left").assign(
        page_views=lambda d: d["page_views"].mask(d["date"] < valid, d["page_views"].fillna(0.0))
    )


def build_series(
    core_path: Path, valid_date: str, freq: str,
    partner_path: Path | None = None, uncensor: bool = False,
    pageviews_path: Path | None = None,
) -> pd.DataFrame:
    valid = pd.Timestamp(valid_date)
    if valid.dayofweek != 6:  # weeks start Sunday; eval assumes aligned valid_date
        raise ValueError(f"valid_date {valid_date} is not a Sunday")

    # target = in-stock units only (OOS days are unobserved demand); future rows
    # censor past-only covariates and promos beyond the known horizon.
    core = _normalize(pd.read_parquet(core_path))
    df = core.assign(
        target=lambda d: d["units"].where((d["oos"].fillna(0) == 0) & (d["date"] < valid)),
        promo_pct_off=lambda d: d["promo_pct_off"].mask(d["date"] >= valid + pd.Timedelta(days=PROMO_KNOWN_DAYS)),
        **core[PAST_ONLY_COLS].mask(core["date"] >= valid, axis=0),
    ).drop(columns=["units"])
    extra_cols = []
    if partner_path is not None:
        df = _add_partner_units(df, partner_path, valid)
        extra_cols.append("partner_units")
    if pageviews_path is not None:
        df = _add_pageviews(df, pageviews_path, valid)
        extra_cols.append("page_views")

    if freq == "D":
        cols = (
            PRODUCT_COLS
            + ["date", "target", "promo_pct_off"]
            + EVENT_COLS
            + PAST_ONLY_COLS
            + extra_cols
        )
        return df[cols]

    offset = (df["date"].dt.dayofweek + 1) % 7
    df["week_start"] = df["date"] - pd.to_timedelta(offset, unit="D")
    agg = {
        "target": ("target", lambda s: s.sum(min_count=1)),
        "in_stock_days": ("target", "count"),
        "week_days": ("date", "count"),
        "promo_pct_off": ("promo_pct_off", "max"),
        "avg_price_paid": ("avg_price_paid", "mean"),
        "buybox_pct": ("buybox_pct", "mean"),
        "buybox_suppression_pct": ("buybox_suppression_pct", "mean"),
        "ad_spend": ("ad_spend", lambda s: s.sum(min_count=1)),
        "oos": ("oos", "mean"),
    } | {c: (c, "max") for c in EVENT_COLS}
    agg |= {c: (c, lambda s: s.sum(min_count=1)) for c in extra_cols}
    wk = df.groupby(PRODUCT_COLS + ["week_start"], as_index=False).agg(**agg)
    if uncensor:
        # Extrapolate observed in-stock demand over listed days in the week.
        wk["target"] *= wk["week_days"] / wk["in_stock_days"].clip(lower=1)
    return wk.drop(columns=["in_stock_days", "week_days"])
