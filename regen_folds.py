"""Residual-stack fold regen — the single source of truth for the residual-stack
training data. `main.py` imports regen_missing_folds() to backfill, before each
forecast, whatever origins its valid_date needs that folds/ doesn't already hold.

For each origin date it generates every member (via src.members.generate_members,
the SAME orchestration the graded main.py uses), then writes folds/<date>.parquet:
the residual-stack training matrix (per-product member forecasts + blend + tvol +
horizon_date), exactly the schema apply_residual_stack reads.

Run from this folder:
  uv run python regen_folds.py --valid-date 2025-12-28   # only the missing origins
  uv run python regen_folds.py [--dates ...]             # explicit origin list
Oldest origin first, so by the time the newest fold lands all its training origins
exist. ~2.5 h/fold on a 5080 → a full 13-origin pass is an overnight job.
"""
import argparse
import time
from pathlib import Path

import pandas as pd

from src.members import generate_members
from src.pipeline import run_pipeline, FM_MEMBERS, BOT50_GBM_MEMBER
from src.core_target import PRODUCT_COLS, load_core_instock
from src.download_data import RAW_DIR, EXTERNAL_DIR, QUERIES, SCHEMA, download_query

INPUT_TABLE = "forecasting_hackathon_core_daily"

HERE = Path(__file__).resolve().parent
FOLD_DIR = HERE / "folds"
CORE = RAW_DIR / "forecasting_hackathon_core_daily.parquet"
HOURLY = EXTERNAL_DIR / "hourly_sales_daily.parquet"
PAGEVIEWS = EXTERNAL_DIR / "page_views_daily.parquet"
KEYS = PRODUCT_COLS + ["horizon_week", "horizon_date"]
# per-product members that become stack features (FM/GRU + log GBM); the reconcile
# aggregates are per-group, excluded
STACK_MEMBER_COLS = [n for n, _ in FM_MEMBERS] + [BOT50_GBM_MEMBER]

# 13 monthly origins ending 2025-10-26 (snapshot supports valid <= 2025-10-26: the
# daily FM members need 91 future core rows and core ends ~2026-01-24). The 5 newest
# each get >=8 prior training origins under the embargo (10-26 -> 12).
DEFAULT_DATES = [(pd.Timestamp("2025-10-26") - pd.Timedelta(weeks=4 * i)).date().isoformat()
                 for i in range(13)][::-1]  # oldest first


def ensure_data(input_table: str = INPUT_TABLE) -> None:
    """Download core_daily + externals via the deliverable's OWN queries (so column
    names match the build_series readers by construction). Idempotent — skips files
    already present."""
    fmt = {"core_table": f"{SCHEMA}.{input_table}"}
    targets = [
        (CORE, QUERIES["core_daily"].format(**fmt)),
        (HOURLY, QUERIES["hourly_sales_daily"]),
        (PAGEVIEWS, QUERIES["page_views_daily"].format(**fmt)),
        (EXTERNAL_DIR / "jbe_page_nodes.parquet", QUERIES["jbe_page_nodes"].format(**fmt)),
    ]
    for path, query in targets:
        if path.exists():
            print(f"data ok: {path}", flush=True)
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"downloading {path} ...", flush=True)
        download_query(query, path)
        print(f"  -> done {path}", flush=True)


def _trailing_vol(valid_date: str) -> pd.DataFrame:
    v = pd.Timestamp(valid_date)
    d = load_core_instock(CORE, valid_date)
    w = d[d["date"] >= v - pd.Timedelta(weeks=12)]
    return w.groupby(PRODUCT_COLS)["units"].sum().rename("tvol").reset_index()


def _fold_matrix(members: dict, blend: pd.DataFrame, valid_date: str) -> pd.DataFrame:
    """Wide per-product matrix: each stack member's forecast + blend + tvol."""
    feats = None
    for name in STACK_MEMBER_COLS:
        if name not in members:
            continue
        m = members[name][KEYS + ["forecast"]].rename(columns={"forecast": name})
        m["page_id"] = m["page_id"].astype(str)
        feats = m if feats is None else feats.merge(m, on=KEYS, how="outer")
    b = blend[KEYS + ["forecast"]].rename(columns={"forecast": "blend"})
    b["page_id"] = b["page_id"].astype(str)
    feats = feats.merge(b, on=KEYS, how="outer")
    feats = feats.merge(_trailing_vol(valid_date), on=PRODUCT_COLS, how="left")
    feats["tvol"] = feats["tvol"].fillna(0.0)
    return feats


def regen_one(date: str, seqgru_seeds: int, num_steps: int) -> None:
    t0 = time.time()
    print(f"\n########## REGEN {date} (seeds={seqgru_seeds}, steps={num_steps}) ##########", flush=True)
    members = generate_members(CORE, date, HOURLY, PAGEVIEWS,
                               seqgru_seeds=seqgru_seeds, num_steps=num_steps)
    # deployment-consistent blend (full promo visibility, like a real historical run)
    blend = run_pipeline(date, members)
    FOLD_DIR.mkdir(parents=True, exist_ok=True)
    _fold_matrix(members, blend, date).to_parquet(FOLD_DIR / f"{date}.parquet", index=False)
    print(f"########## {date} done in {(time.time()-t0)/60:.1f} min "
          f"({len(members)} members) ##########", flush=True)


def missing_fold_dates(valid_date: str, step_weeks: int = 4) -> list[str]:
    """Monthly fold origins before valid_date that folds/ doesn't have yet, stepping
    forward from the newest existing fold. Empty when folds/ is current (or empty --
    ship folds/ or pass --dates for a cold build)."""
    existing = {p.stem for p in FOLD_DIR.glob("*.parquet")}
    if not existing:
        return []
    v, step = pd.Timestamp(valid_date), pd.Timedelta(weeks=step_weeks)
    out, d = [], pd.Timestamp(max(existing)) + step
    while d < v:
        if d.date().isoformat() not in existing:
            out.append(d.date().isoformat())
        d += step
    return out


def regen_missing_folds(valid_date: str, seqgru_seeds: int = 3, num_steps: int = 8000,
                        input_table: str = INPUT_TABLE) -> None:
    """Snapshot data, then build the folds this valid_date needs that folds/ lacks.
    In production folds accumulate across runs; a cold competition date backfills
    them here, before the forecast. Idempotent -- no-op when folds/ is current."""
    ensure_data(input_table)
    dates = missing_fold_dates(valid_date)
    print(f"folds: backfilling {dates}" if dates else f"folds: current for {valid_date}", flush=True)
    for d in dates:
        regen_one(d, seqgru_seeds, num_steps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dates", nargs="*", default=DEFAULT_DATES)
    ap.add_argument("--valid-date", default=None,
                    help="build only the folds a run at this valid_date is missing "
                         "(overrides --dates); the same backfill main.py does")
    ap.add_argument("--seqgru-seeds", type=int, default=3)
    ap.add_argument("--num-steps", type=int, default=8000)
    ap.add_argument("--skip-existing", action="store_true",
                    help="skip dates whose folds/<date>.parquet already exists")
    a = ap.parse_args()
    ensure_data()
    dates = missing_fold_dates(a.valid_date) if a.valid_date else a.dates
    print(f"regen {len(dates)} folds: {dates}", flush=True)
    t0 = time.time()
    failed = []
    for d in dates:
        if a.skip_existing and (FOLD_DIR / f"{d}.parquet").exists():
            print(f"skip {d} (exists)", flush=True)
            continue
        try:
            regen_one(d, a.seqgru_seeds, a.num_steps)
        except Exception as e:  # one bad fold shouldn't abort an overnight run
            import traceback
            traceback.print_exc()
            print(f"!!! fold {d} FAILED: {e} — continuing", flush=True)
            failed.append(d)
    print(f"\nALL DONE in {(time.time()-t0)/60:.1f} min"
          + (f" — FAILED folds: {failed}" if failed else ""), flush=True)


if __name__ == "__main__":
    main()
