"""Residual stacking: a regularized L1 LightGBM predicts the blend's residual
(in-stock actual - blend) from the member forecasts + blend + horizon + trailing
volume, added back to the pipeline output.

Trains on the folds/ matrices (one wide matrix per origin: member forecasts +
blend + trailing volume, all computed using only data before that origin). The
target is in-stock weekly units from core_daily, LEAK-GUARDED by a single date
cutoff: `_weekly_target` keeps only weeks strictly before valid_date (the real
forecast date, via load_core_instock(date < valid_date)), and each fold is joined
to it on horizon_date. So a fold whose 13-week horizon straddles valid_date
contributes only its pre-valid_date weeks; nothing from on/after valid_date can
enter training. The cutoff is the actual week date, never the fold's filename.

The target is the RESIDUAL (small; the model can safely predict ~0), the objective
is L1 (WAPE-aligned), and the model is heavily regularized. Cold-start rows (no
member covered them) keep their pipeline value -- the stack has no signal there.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

from .core_target import CORE_DAILY_PATH, PRODUCT_COLS, load_core_instock
from .pipeline import FM_MEMBERS

FOLD_DIR = Path(__file__).resolve().parent.parent / "folds"
KEYS = PRODUCT_COLS + ["horizon_week", "horizon_date"]
# per-product members only (FM/GRU members + log GBM); reconcile aggregates are
# per-group (page_id=_AGG_), all-NaN per product, so not stack inputs
MEMBER_COLS = [n for n, _ in FM_MEMBERS] + ["gbm_log"]
# deterministic+seeded (LightGBM is otherwise nondeterministic run-to-run); deepened
# as backtest origins grew (5 folds -> nl31, 8 folds -> nl63/mcs300). See scripts/.
STACK_PARAMS = dict(objective="regression_l1", num_leaves=63, min_child_samples=300,
                    lambda_l2=3.0, learning_rate=0.03, feature_fraction=0.7, verbose=-1,
                    seed=42, deterministic=True, force_row_wise=True, num_threads=8)
STACK_ROUNDS = 700


def _weekly_target(valid_date: str, core_path: Path) -> pd.DataFrame:
    """In-stock Sunday-week units for weeks strictly before valid_date -- the only
    weeks known at the real forecast time. load_core_instock clamps to date <
    valid_date (the actual date), so no fold's post-valid_date horizon can leak in."""
    d = load_core_instock(core_path, valid_date)
    off = (d["date"].dt.dayofweek + 1) % 7
    d = d.assign(horizon_date=d["date"] - pd.to_timedelta(off, unit="D"))
    return d.groupby(PRODUCT_COLS + ["horizon_date"])["units"].sum().rename("y").reset_index()


def _trailing_vol(valid_date: str, core_path: Path) -> pd.DataFrame:
    v = pd.Timestamp(valid_date)
    d = load_core_instock(core_path, valid_date)
    w = d[d["date"] >= v - pd.Timedelta(weeks=12)]
    return w.groupby(PRODUCT_COLS)["units"].sum().rename("tvol").reset_index()


def _members_wide(members: dict[str, pd.DataFrame]) -> pd.DataFrame:
    feats = None
    for name in MEMBER_COLS:
        if name not in members:
            continue
        m = members[name][KEYS + ["forecast"]].rename(columns={"forecast": name})
        m["page_id"] = m["page_id"].astype(str)
        feats = m if feats is None else feats.merge(m, on=KEYS, how="outer")
    for c in MEMBER_COLS:
        if c not in feats:
            feats[c] = np.nan
    return feats


def apply_residual_stack(out: pd.DataFrame, members: dict[str, pd.DataFrame],
                         valid_date: str, core_path: Path = CORE_DAILY_PATH) -> pd.DataFrame:
    """Add the residual-stack correction to `out` (the pipeline output). Trains on
    every fold's weeks before valid_date; no-op if folds/ is empty."""
    featcols = MEMBER_COLS + ["blend", "horizon_week", "tvol"]
    # weeks strictly before valid_date; the inner-join below drops each fold's
    # on/after-valid_date horizon, so it is the whole leak guard. Glob at call time
    # (not import) so folds main.py backfilled this run are picked up.
    target = _weekly_target(valid_date, core_path)
    rows = []
    for fp in sorted(FOLD_DIR.glob("*.parquet")) if FOLD_DIR.exists() else []:
        m = pd.read_parquet(fp)
        m["page_id"] = m["page_id"].astype(str)
        m = m.merge(target, on=PRODUCT_COLS + ["horizon_date"], how="inner")
        if len(m):
            m["resid"] = m["y"] - m["blend"]
            rows.append(m)
    if not rows:
        print("residual_stack: no fold weeks before valid_date — skipping")
        return out
    tr = pd.concat(rows, ignore_index=True)
    model = lgb.train(STACK_PARAMS, lgb.Dataset(tr[featcols], label=tr["resid"]),
                      num_boost_round=STACK_ROUNDS)

    feats = _members_wide(members)
    o = out.copy()
    o["page_id"] = o["page_id"].astype(str)
    feats = feats.merge(o[KEYS + ["forecast"]].rename(columns={"forecast": "blend"}),
                        on=KEYS, how="right")
    feats = feats.merge(_trailing_vol(valid_date, core_path), on=PRODUCT_COLS, how="left")
    feats["tvol"] = feats["tvol"].fillna(0.0)
    adj = np.clip(feats["blend"] + model.predict(feats[featcols]), 0, None)
    covered = feats[MEMBER_COLS].notna().any(axis=1)
    feats["adj"] = np.where(covered, adj, feats["blend"])
    o = o.merge(feats[KEYS + ["adj"]], on=KEYS, how="left")
    o["forecast"] = o["adj"].fillna(o["forecast"])
    return o.drop(columns=["adj"])
