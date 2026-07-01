"""Blend the ensemble members and post-process to the final forecast.

Chain: fixed-weight blend -> event-conditional GBM upweight on Black-Friday /
Cyber-Monday horizons -> Christmas/NY seasonal correction -> promo reallocation
-> horizon-ramped level uplift -> cold-start fill -> bottom-50% log-GBM routing
-> hierarchical reconciliation to partner aggregates -> Q4 event uplift.
"""
import pandas as pd

from .blend import blend
from .core_target import PRODUCT_COLS
from .promo_realloc import CORE_DAILY_PATH, promo_reallocate
from .seasonal_adjust import _week_events, seasonal_factors

FM_MEMBERS = [
    ("chronos2_ft8000pcov_pv_D_q0.55", 0.120),
    ("chronos2_ft8000pcov_pv_D_q0.5", 0.400),
    ("chronos2_ft8000unc_W_q0.55", 0.020),
    ("chronos2_ft8000unc_W_q0.6", 0.080),
    ("timesfm25_ft8000unc_W_q0.6", 0.050),
    ("timesfm25_zeroshot_W_unc_q0.7", 0.060),
    ("timesfm25_zeroshot_D_q0.6", 0.040),
    ("gbm_cat", 0.180),
    ("seqgru_W_q0.55", 0.130),  # directly-trained GRU member
]
EVENT_GBM_ALPHA = 0.25
LEVEL_RAMP = 0.02
BOT50_GBM_MEMBER = "gbm_log"
BOT50_GBM_WEIGHT = 0.4  # stronger base bot50 tier needs less log-GBM correction
# reconcile toward a 50/50 blend of the weekly + daily partner x marketplace aggregates
RECONCILE_MEMBERS = [("reconcile_agg", 0.5), ("reconcile_agg_D", 0.5)]
RECONCILE_LAM = 0.3
RECONCILE_CLIP = (0.7, 1.4)
EVENT_UPLIFT = {"black_fri": 1.04, "pre_black_fri": 1.03, "cyber_mon": 1.04}  # BF/pre-BF/CM week lift


def _bot50_products(valid_date: str) -> pd.DataFrame:
    """Low-volume tier: products with trailing-12w in-stock units (core, oos==0,
    < valid_date) at or below the median — the eval's bottom-50% set, from core."""
    valid = pd.Timestamp(valid_date)
    core = pd.read_parquet(CORE_DAILY_PATH, columns=PRODUCT_COLS + ["date", "units", "oos"])
    core["date"] = pd.to_datetime(core["date"])
    win = core[(core["date"] >= valid - pd.Timedelta(weeks=12)) & (core["date"] < valid)
               & (core["oos"].fillna(0) == 0)]
    s = win.groupby(PRODUCT_COLS)["units"].sum()
    return (s[s <= s.quantile(0.50)].index.to_frame(index=False)
            .astype({"marketplace_id": int, "partner_id": int, "page_id": str}))


def _complete_universe(out: pd.DataFrame, valid_date: str) -> pd.DataFrame:
    """One row per horizon week for every product with core rows in the window;
    products a member skipped get 0, products absent from the window stay out."""
    core = pd.read_parquet(CORE_DAILY_PATH, columns=PRODUCT_COLS + ["date"])
    core["date"] = pd.to_datetime(core["date"])
    valid = pd.Timestamp(valid_date)
    hmax = int(out["horizon_week"].max())
    in_window = (core["date"] >= valid) & (core["date"] < valid + pd.Timedelta(days=7 * hmax))
    universe = (core.loc[in_window, PRODUCT_COLS].drop_duplicates()
                .astype({"marketplace_id": int, "partner_id": int, "page_id": str}))
    horizons = pd.DataFrame({"horizon_week": range(1, hmax + 1)})
    horizons["horizon_date"] = valid + pd.to_timedelta(7 * (horizons["horizon_week"] - 1), unit="D")
    keys = PRODUCT_COLS + ["horizon_week", "horizon_date"]
    merged = universe.merge(horizons, how="cross").merge(out, on=keys, how="outer")
    missing = int(merged["forecast"].isna().sum())
    if missing:
        print(f"cold-start fill: {missing:,} rows with no member forecast set to 0")
    merged["forecast"] = merged["forecast"].fillna(0.0)
    return merged


def reconcile_to_groups(out: pd.DataFrame, aggs: list[tuple[pd.DataFrame, float]],
                        lam: float = RECONCILE_LAM,
                        clip: tuple[float, float] = RECONCILE_CLIP) -> pd.DataFrame:
    """Scale each product toward its partner x marketplace group total: ONE damped,
    clipped factor per group from horizon-summed totals (corrects group level,
    preserving each product's horizon shape). `aggs` is a list of (forecast_df,
    weight) blended into the target; None/absent entries drop out and renormalize."""
    aggs = [(df, w) for df, w in aggs if df is not None]
    if not lam or not aggs:
        return out
    total = sum(w for _, w in aggs)
    gk = ["marketplace_id", "partner_id"]
    agg = (blend([(df, w / total) for df, w in aggs])
           .groupby(gk, as_index=False)["forecast"].sum()
           .rename(columns={"forecast": "_agg"}))
    g = out.groupby(gk, as_index=False)["forecast"].sum().rename(columns={"forecast": "_g"})
    ga = g.merge(agg, on=gk, how="left")
    ga["_ratio"] = (ga["_agg"] / ga["_g"]).where(ga["_g"] > 0)
    ga["_factor"] = (ga["_ratio"].clip(*clip) ** lam).fillna(1.0)
    out = out.merge(ga[gk + ["_factor"]], on=gk, how="left")
    out["forecast"] = out["forecast"] * out["_factor"].fillna(1.0)
    return out.drop(columns=["_factor"])


def event_uplift(out: pd.DataFrame, valid_date: str,
                 uplift: dict[str, float] = EVENT_UPLIFT) -> pd.DataFrame:
    """Small multiplicative level lift on the BF / pre-BF / CM calendar weeks (residual
    group-level Q4 under-forecast). No-op off those weeks."""
    if not uplift:
        return out
    valid = pd.Timestamp(valid_date)
    factors: dict[int, float] = {}
    bf_h = None
    for h in out["horizon_week"].unique():
        ev = _week_events(valid + pd.Timedelta(weeks=int(h) - 1)).keys()
        f = 1.0
        if "black_fri" in ev:
            f *= uplift.get("black_fri", 1.0)
            bf_h = int(h)
        if "cyber_mon" in ev:
            f *= uplift.get("cyber_mon", 1.0)
        if f != 1.0:
            factors[int(h)] = f
    if bf_h is not None and "pre_black_fri" in uplift:
        pre = bf_h - 1
        factors[pre] = factors.get(pre, 1.0) * uplift["pre_black_fri"]
    if not factors:
        return out
    out = out.copy()
    out["forecast"] = out["forecast"] * out["horizon_week"].map(factors).fillna(1.0)
    return out


def run_pipeline(
    valid_date: str,
    members: dict[str, pd.DataFrame],
    promo_known_days: int | None = None,
    event_gbm_alpha: float = EVENT_GBM_ALPHA,
    level_ramp: float = LEVEL_RAMP,
    bot50_gbm_weight: float = BOT50_GBM_WEIGHT,
    reconcile_lam: float = RECONCILE_LAM,
) -> pd.DataFrame:
    present = [(name, w) for name, w in FM_MEMBERS if name in members]
    total = sum(w for _, w in present)
    fm = blend([(members[name], w / total) for name, w in present])

    if event_gbm_alpha:
        gbm_name = next((n for n in members if "gbm" in n), None)
        valid = pd.Timestamp(valid_date)
        event_hs = [h for h in fm["horizon_week"].unique()
                    if {"black_fri", "cyber_mon"}
                    & _week_events(valid + pd.Timedelta(weeks=int(h) - 1)).keys()]
        if gbm_name and event_hs:
            gbm = blend([(members[gbm_name], 1.0)]).rename(columns={"forecast": "_gbm"})
            fm = fm.merge(gbm, on=[c for c in fm.columns if c != "forecast"], how="left")
            m = fm["horizon_week"].isin(event_hs) & fm["_gbm"].notna()
            fm.loc[m, "forecast"] = ((1 - event_gbm_alpha) * fm.loc[m, "forecast"]
                                     + event_gbm_alpha * fm.loc[m, "_gbm"])
            fm = fm.drop(columns=["_gbm"])

    stacked = fm.copy()
    factors = seasonal_factors(stacked, valid_date)
    stacked["forecast"] = stacked["forecast"] * stacked["horizon_week"].map(factors)
    stacked = promo_reallocate(stacked, valid_date, promo_known_days=promo_known_days)

    if level_ramp:
        stacked = stacked.copy()
        h = stacked["horizon_week"].to_numpy()
        ramp = 1.0 + level_ramp * (h - 1) / max(h.max() - 1, 1)
        stacked["forecast"] = stacked["forecast"] * ramp

    if bot50_gbm_weight and BOT50_GBM_MEMBER in members:
        keys = [c for c in stacked.columns if c != "forecast"]
        gbm = blend([(members[BOT50_GBM_MEMBER], 1.0)]).rename(columns={"forecast": "_blog"})
        bot = _bot50_products(valid_date)
        bot["_bot"] = 1
        stacked = stacked.merge(gbm, on=keys, how="left").merge(bot, on=PRODUCT_COLS, how="left")
        m = (stacked["_bot"] == 1) & stacked["_blog"].notna()
        stacked.loc[m, "forecast"] = ((1 - bot50_gbm_weight) * stacked.loc[m, "forecast"]
                                      + bot50_gbm_weight * stacked.loc[m, "_blog"])
        stacked = stacked.drop(columns=["_blog", "_bot"])

    out = _complete_universe(stacked, valid_date)
    aggs = [(members.get(name), w) for name, w in RECONCILE_MEMBERS]
    out = reconcile_to_groups(out, aggs, lam=reconcile_lam)
    out = event_uplift(out, valid_date)  # applied last
    return out
