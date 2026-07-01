"""Tweedie LightGBM weekly forecaster: leakage-safe multi-origin training
(origins every 4 weeks before valid_date; features strictly before each origin,
future-promo masked beyond origin+28d). Features: partner/marketplace identity,
trailing means, lags, trend, volatility, covariates, event flags, leaf nodes.
"""

from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from ..core_target import PRODUCT_COLS
from ..promo_realloc import CORE_DAILY_PATH, load_core_daily
from ..seasonal_adjust import _week_events
from .common import to_submission

PAGE_NODES_PATH = Path("data/external/jbe_page_nodes.parquet")

HORIZON_WEEKS = 13
ORIGIN_STEP_WEEKS = 4
N_ORIGINS = 19
TRAIL_WINDOWS = [4, 8, 13, 26, 52]
PROMO_KNOWN_DAYS = 28

PARAMS = dict(
    learning_rate=0.05,
    num_leaves=127,
    min_data_in_leaf=200,
    feature_fraction=0.9,
    bagging_fraction=0.8,
    bagging_freq=1,
    verbosity=-1,
    deterministic=True,
    force_row_wise=True,
)
NUM_ROUNDS = 600


def make_params(objective: str, alpha: float, tweedie_power: float, seed: int) -> dict:
    """Objective-specific LightGBM params."""
    base = dict(PARAMS, seed=seed)
    if objective == "tweedie":
        return dict(base, objective="tweedie", tweedie_variance_power=tweedie_power)
    if objective == "regression":
        return dict(base, objective="regression")
    return dict(base, objective="quantile", alpha=alpha)


def weekly_matrices(valid_date: str):
    """Weekly matrices: in-stock units, promo flag, and past-only covariates."""
    valid = pd.Timestamp(valid_date)
    cd = (
        pd.read_parquet(CORE_DAILY_PATH, columns=PRODUCT_COLS + [
            "date", "units", "promo_pct_off", "avg_price_paid", "buybox_pct", "ad_spend", "oos"])
        .rename(columns=str.lower)
        .astype({"partner_id": int, "marketplace_id": int, "page_id": str})
        .assign(
            date=lambda d: pd.to_datetime(d["date"]),
            week=lambda d: d["date"] - pd.to_timedelta((d["date"].dt.dayofweek + 1) % 7, unit="D"),
        )
    )
    # OOS days are unobserved demand, so labels and lag features use in-stock days.
    instock = cd[(cd["date"] < valid) & (cd["oos"].fillna(0) == 0)]
    units = instock.pivot_table(index=PRODUCT_COLS, columns="week", values="units",
                                aggfunc="sum")
    promo = cd.assign(p=(cd["promo_pct_off"].fillna(0) > 0).astype(float)) \
        .pivot_table(index=PRODUCT_COLS, columns="week", values="p", aggfunc="max")
    promo_depth = cd.pivot_table(index=PRODUCT_COLS, columns="week",
                                 values="promo_pct_off", aggfunc="mean")
    past = cd[cd["date"] < valid]
    covs = {
        name: past.pivot_table(index=PRODUCT_COLS, columns="week", values=col,
                               aggfunc="mean")
        for name, col in [("price", "avg_price_paid"), ("buybox", "buybox_pct"),
                          ("adspend", "ad_spend"), ("oos", "oos")]
    }

    weeks = pd.date_range(units.columns.min(),
                          valid + pd.Timedelta(weeks=HORIZON_WEEKS - 1), freq="7D")
    units = units.reindex(columns=weeks)
    promo = promo.reindex(index=units.index, columns=weeks).fillna(0.0)
    promo_depth = promo_depth.reindex(index=units.index, columns=weeks).fillna(0.0)
    covs = {k: v.reindex(index=units.index, columns=weeks) for k, v in covs.items()}
    return units, promo, promo_depth, covs


def build_rows(units: pd.DataFrame, promo: pd.DataFrame, promo_depth: pd.DataFrame,
               covs: dict, origins: list[int], valid_idx: int, for_training: bool,
               rich: bool = False):
    """Feature/target rows for the given origin column-indices.

    rich adds past-only EWMA + rolling median/min/max."""
    weeks = units.columns
    u = units.to_numpy()
    obs = ~np.isnan(u)
    u0 = np.nan_to_num(u)
    csum = np.cumsum(u0, axis=1)
    csum2 = np.cumsum(u0 ** 2, axis=1)
    ccnt = np.cumsum(obs, axis=1)
    promo_np = promo.to_numpy()
    promo_depth_np = promo_depth.to_numpy()
    covs_np = {k: v.to_numpy() for k, v in covs.items()}

    def window_mean(o, w):
        lo = max(o - w, 0)
        s = csum[:, o - 1] - (csum[:, lo - 1] if lo > 0 else 0)
        n = ccnt[:, o - 1] - (ccnt[:, lo - 1] if lo > 0 else 0)
        return np.divide(s, n, out=np.zeros_like(s), where=n > 0), n

    frames = []
    for o in origins:
        feats = {}
        for w in TRAIL_WINDOWS:
            feats[f"mean{w}"], n = window_mean(o, w)
            if w == 8:
                feats["obs8"] = n
        # weeks since last observed sale (>0 units) and since first observation
        seen = obs[:, :o]
        any_seen = seen.any(axis=1)
        first = np.where(any_seen, seen.argmax(axis=1), o)
        feats["age_w"] = o - first
        sold = (u0[:, :o] > 0)
        any_sold = sold.any(axis=1)
        last_sold = o - 1 - np.where(any_sold, sold[:, ::-1].argmax(axis=1), o - 1)
        feats["wk_since_sale"] = np.where(any_sold, o - 1 - last_sold, 999)
        # recent individual lags (raw weekly units, NaN if that week unobserved
        # -> LightGBM handles NaN): momentum the trailing means smooth over
        for k in (1, 2, 4, 8, 13):
            feats[f"lag{k}"] = u[:, o - k]
        # recent-vs-baseline trend and volatility (coeff. of variation, 13w)
        feats["trend"] = feats["mean4"] / (feats["mean13"] + 1e-6)
        vlo = max(o - 13, 0)
        sm = csum[:, o - 1] - (csum[:, vlo - 1] if vlo > 0 else 0)
        sq = csum2[:, o - 1] - (csum2[:, vlo - 1] if vlo > 0 else 0)
        nn = ccnt[:, o - 1] - (ccnt[:, vlo - 1] if vlo > 0 else 0)
        m13 = np.divide(sm, nn, out=np.zeros_like(sm), where=nn > 0)
        var13 = np.divide(sq, nn, out=np.zeros_like(sq), where=nn > 0) - m13 ** 2
        feats["cv13"] = np.sqrt(np.clip(var13, 0, None)) / (m13 + 1e-6)
        for name, mat in covs_np.items():
            with np.errstate(invalid="ignore"):
                feats[f"{name}8"] = np.nanmean(mat[:, max(o - 8, 0):o], axis=1)

        if rich:
            win13 = u[:, max(o - 13, 0):o]
            with np.errstate(invalid="ignore"):
                feats["med13"] = np.nanmedian(win13, axis=1)
                feats["min13"] = np.nanmin(win13, axis=1)
                feats["max13"] = np.nanmax(win13, axis=1)
            win26 = u[:, max(o - 26, 0):o]
            ages = np.arange(win26.shape[1])[::-1]
            wts = 0.85 ** ages
            m = ~np.isnan(win26)
            num = np.nansum(np.where(m, win26, 0.0) * wts, axis=1)
            den = (m * wts).sum(axis=1)
            feats["ewma"] = np.divide(num, den, out=np.zeros_like(num), where=den > 0)

        base = pd.DataFrame(feats, index=units.index)
        for h in range(1, HORIZON_WEEKS + 1):
            t = o + h - 1
            if t >= len(weeks):
                continue
            if for_training and t >= valid_idx:
                continue
            row = base.copy()
            tgt_week = weeks[t]
            row["horizon"] = h
            row["woy"] = tgt_week.isocalendar().week
            ev = _week_events(tgt_week)
            row["ev_bfcm"] = float(bool({"black_fri", "cyber_mon"} & ev.keys()))
            row["ev_xmas"] = float("christmas" in ev)
            row["ev_ny"] = float("ny_day" in ev)
            # Signed weeks to nearest Christmas captures pre/post-holiday shape.
            xmas = min((pd.Timestamp(tgt_week.year + dy, 12, 25) for dy in (-1, 0, 1)),
                       key=lambda c: abs((tgt_week - c).days))
            row["wks_to_xmas"] = (tgt_week - xmas).days / 7.0
            # lag-52: same week last year (mean of +-1 week, observed only)
            lo52 = max(t - 53, 0)
            s52 = csum[:, min(t - 51, len(weeks) - 1)] - (csum[:, lo52 - 1] if lo52 > 0 else 0)
            n52 = ccnt[:, min(t - 51, len(weeks) - 1)] - (ccnt[:, lo52 - 1] if lo52 > 0 else 0)
            row["lag52"] = np.divide(s52, n52, out=np.full_like(s52, np.nan), where=n52 > 0)
            # future promo, masked to known horizon (binary flag + discount depth)
            known = (tgt_week - weeks[o]).days <= PROMO_KNOWN_DAYS
            row["promo_fut"] = promo_np[:, t] if known else 0.0
            row["promo8"] = promo_np[:, max(o - 8, 0):o].mean(axis=1)
            row["promo_depth_fut"] = promo_depth_np[:, t] if known else 0.0
            row["promo_depth8"] = promo_depth_np[:, max(o - 8, 0):o].mean(axis=1)
            if for_training:
                row["y"] = u[:, t]
            row["origin"] = o
            frames.append(row.reset_index())
    df = pd.concat(frames, ignore_index=True)
    if for_training:
        df = df.dropna(subset=["y"])
    return df


def load_page_leaf_nodes(path: Path = PAGE_NODES_PATH) -> pd.DataFrame:
    """One leaf (deepest) browse-node per page, as integer codes for LightGBM
    categorical use. A page can sit on several browse paths; we keep the
    deepest node (finest category). Pages absent here (non-US, uncatalogued)
    get -1 = LightGBM's missing-category bin. Codes are global (built from the
    full file), so train and predict frames share the same encoding."""
    df = (
        pd.read_parquet(path)[["page_id", "browse_node_id", "depth"]]
        .sort_values("depth")
        .drop_duplicates("page_id", keep="last")
    )
    codes = {n: i for i, n in enumerate(sorted(df["browse_node_id"].unique()))}
    return df.assign(
        node=lambda d: d["browse_node_id"].map(codes).astype("int32"),
        page_id=lambda d: d["page_id"].astype(str),
    )[["page_id", "node"]]


def run_gbm(valid_date, objective="tweedie", alpha=0.6, tweedie_power=1.1,
            seed=42, category_nodes=True, rich=True, log_target=False) -> pd.DataFrame:
    """Train a LightGBM member and return its forecast (submission schema).

    The bottom-volume specialist passes log_target=True."""
    units, promo, promo_depth, covs = weekly_matrices(valid_date)
    weeks = units.columns
    valid_idx = weeks.get_loc(pd.Timestamp(valid_date))

    train_origins = [valid_idx - k * ORIGIN_STEP_WEEKS for k in range(1, N_ORIGINS + 1)]
    train_origins = [o for o in train_origins if o >= 60]
    train = build_rows(units, promo, promo_depth, covs, train_origins, valid_idx,
                       for_training=True, rich=rich)
    print(f"train rows: {len(train):,} from {len(train_origins)} origins")

    # partner_id + marketplace_id as categoricals give the model product/partner
    # identity (page_id stays out — 24k levels would overfit); LightGBM bins them
    cat_feats = ["partner_id", "marketplace_id"]
    nodes = None
    if category_nodes and not PAGE_NODES_PATH.exists():
        print(f"WARNING: {PAGE_NODES_PATH} absent — training without category nodes")
    elif category_nodes:
        nodes = load_page_leaf_nodes()
        train = train.merge(nodes, on="page_id", how="left")
        train["node"] = train["node"].fillna(-1).astype("int32")
        cat_feats = cat_feats + ["node"]
    feat_cols = [c for c in train.columns if c not in ["page_id", "y", "origin"]]
    y = train["y"].to_numpy()
    label = np.log1p(y) if log_target else y
    dtrain = lgb.Dataset(train[feat_cols], label=label,
                         categorical_feature=cat_feats)

    # predict only the forecast universe (core_daily products); the actuals
    # panel has ~3x more products, which are useful for training only
    universe = load_core_daily(CORE_DAILY_PATH)[PRODUCT_COLS].drop_duplicates()
    pred = build_rows(units, promo, promo_depth, covs, [valid_idx], valid_idx,
                      for_training=False, rich=rich)
    pred = pred.merge(universe, on=PRODUCT_COLS)
    if nodes is not None:
        pred = pred.merge(nodes, on="page_id", how="left")
        pred["node"] = pred["node"].fillna(-1).astype("int32")

    obj = "regression" if log_target else objective
    params = make_params(obj, alpha, tweedie_power, seed)
    model = lgb.train(params, dtrain, num_boost_round=NUM_ROUNDS)
    raw = model.predict(pred[feat_cols])
    if log_target:
        raw = np.expm1(raw)
    pred["forecast"] = raw  # to_submission clips negatives
    pred["horizon_date"] = pd.Timestamp(valid_date) + pd.to_timedelta((pred["horizon"] - 1) * 7, unit="D")
    return to_submission(pred, valid_date)


if __name__ == "__main__":
    # Run as a subprocess (python -m src.models.gbm) so LightGBM's OpenMP runtime
    # doesn't share a process with torch's (segfaults on macOS). The two flag
    # combinations reproduce main.py's gbm_cat and gbm_log members.
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--valid-date", required=True)
    ap.add_argument("--log-target", action="store_true")
    ap.add_argument("-o", "--output", type=Path, required=True)
    args = ap.parse_args()
    run_gbm(args.valid_date, log_target=args.log_target,
            rich=not args.log_target).to_parquet(args.output, index=False)
