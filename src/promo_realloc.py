"""Within Black Friday / Cyber Monday weeks, reallocate forecast volume toward
planned-promo products (promo_pct_off > 0 on future core_daily rows). Promo
products are scaled toward the prior-year analog week's promo/non-promo lift
ratio (de-double-counted by the ratio the forecast already implies), then the
week is rescaled to its original total — pure reallocation, weekly level intact.
"""

from pathlib import Path

import pandas as pd

from .core_target import PRODUCT_COLS
from .seasonal_adjust import _week_events
CORE_DAILY_PATH = Path("data/raw/forecasting_hackathon_core_daily.parquet")
MIN_PROMO_PRODUCTS = 1000
MIN_BASELINE = 5
RATIO_CLIP = (0.8, 2.0)
MIN_BOOST_DELTA = 0.1


def load_core_daily(path: Path = CORE_DAILY_PATH) -> pd.DataFrame:
    return (pd.read_parquet(path, columns=PRODUCT_COLS + ["date", "units", "promo_pct_off"])
            .rename(columns=str.lower)
            .astype({"marketplace_id": int, "partner_id": int, "page_id": str})
            .assign(date=lambda d: pd.to_datetime(d["date"])))


def weekly(df: pd.DataFrame) -> pd.DataFrame:
    offset = (df["date"].dt.dayofweek + 1) % 7
    out = df.assign(week=df["date"] - pd.to_timedelta(offset, unit="D"))
    return out.groupby(PRODUCT_COLS + ["week"]).agg(
        units=("units", "sum"),
        promo=("promo_pct_off", lambda x: x.fillna(0).max()),
    ).reset_index()


def target_ratio(wk: pd.DataFrame, event_week: pd.Timestamp) -> float | None:
    """Vol-weighted promo/non-promo lift at event_week vs 4-week pre-event baseline."""
    basew = [event_week - pd.Timedelta(weeks=k) for k in range(2, 6)]
    sub = wk[wk["week"].isin(basew + [event_week])]
    pu = sub.pivot_table(index=PRODUCT_COLS, columns="week", values="units")
    pp = sub.pivot_table(index=PRODUCT_COLS, columns="week", values="promo")
    if event_week not in pu.columns:
        return None
    d = pd.DataFrame(
        {"base": pu[[w for w in basew if w in pu.columns]].mean(axis=1),
         "ev": pu[event_week], "promo": pp.get(event_week)}
    ).dropna(subset=["base", "ev"])
    d = d[d["base"] >= MIN_BASELINE]
    promo = d[d["promo"].fillna(0) > 0]
    nonpromo = d[d["promo"].fillna(0) == 0]
    if len(promo) < 100 or nonpromo["base"].sum() == 0 or promo["base"].sum() == 0:
        return None
    return (promo["ev"].sum() / promo["base"].sum()) / (
        nonpromo["ev"].sum() / nonpromo["base"].sum()
    )


def promo_reallocate(
    forecast: pd.DataFrame,
    valid_date: str,
    core_daily_path: Path = CORE_DAILY_PATH,
    promo_known_days: int | None = None,
) -> pd.DataFrame:
    """If promo_known_days is set, mask promo rows after valid_date + N days."""
    valid = pd.Timestamp(valid_date)
    cd = load_core_daily(core_daily_path)
    fut = cd[cd["date"] >= valid]
    if promo_known_days is not None:
        fut = fut.copy()
        fut.loc[
            fut["date"] > valid + pd.Timedelta(days=promo_known_days), "promo_pct_off"
        ] = 0.0
    future_wk = weekly(fut)
    hist_wk = weekly(cd[cd["date"] < valid])

    out = forecast.copy()
    prod_tuples = pd.Series(
        list(map(tuple, out[PRODUCT_COLS].to_numpy())), index=out.index
    )
    base_fcst = out[out["horizon_week"].isin([2, 3, 4])].groupby(PRODUCT_COLS)[
        "forecast"
    ].mean()

    for h in sorted(out["horizon_week"].unique()):
        week_start = valid + pd.Timedelta(weeks=h - 1)
        if not {"black_fri", "cyber_mon"} & _week_events(week_start).keys():
            continue
        fw = future_wk[future_wk["week"] == week_start]
        promo_set = set(map(tuple, fw[fw["promo"] > 0][PRODUCT_COLS].to_numpy()))
        if len(promo_set) < MIN_PROMO_PRODUCTS:
            continue
        target = target_ratio(hist_wk, week_start - pd.Timedelta(days=364))
        if target is None:
            continue

        wk_f = out[out["horizon_week"] == h].set_index(PRODUCT_COLS)["forecast"]
        d = pd.concat({"ev": wk_f, "base": base_fcst}, axis=1).dropna()
        d = d[d["base"] > 0.5]
        is_promo = d.index.isin(promo_set)
        if d[is_promo]["base"].sum() == 0 or d[~is_promo]["base"].sum() == 0:
            continue
        implied = (d[is_promo]["ev"].sum() / d[is_promo]["base"].sum()) / (
            d[~is_promo]["ev"].sum() / d[~is_promo]["base"].sum()
        )

        boost = min(max(target / implied, RATIO_CLIP[0]), RATIO_CLIP[1])
        if abs(boost - 1) <= MIN_BOOST_DELTA:
            continue
        mask_h = out["horizon_week"] == h
        total = out.loc[mask_h, "forecast"].sum()
        out.loc[mask_h & prod_tuples.isin(promo_set), "forecast"] *= boost
        out.loc[mask_h, "forecast"] *= total / out.loc[mask_h, "forecast"].sum()
        print(
            f"H{h} ({week_start.date()}): {len(promo_set):,} promo products, "
            f"target {target:.3f}, implied {implied:.3f}, boost {boost:.3f}"
        )
    return out
