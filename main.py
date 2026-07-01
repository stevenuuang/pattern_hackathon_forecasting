"""13-week unit-sales forecast deliverable.

Run from this folder:  uv sync && uv run python main.py
The forecast is ~2.5 h on an RTX 5080 (three 8000-step fine-tunes dominate). If the
residual-stack folds this valid_date needs already exist (the production case), that
is the whole run; a cold date backfills its missing folds first, ~2.5 h each.
"""
import time

from regen_folds import ensure_data, regen_missing_folds
from src.download_data import EXTERNAL_DIR, RAW_DIR
from src.members import generate_members
from src.pipeline import run_pipeline
from src.residual_stack import apply_residual_stack

INPUT_TABLE = "forecasting_hackathon_core_daily"
VALID_DATE  = "2025-10-26"
OUTPUT_CSV  = "submission.csv"
BACKFILL_FOLDS = True  # False -> forecast on the folds already in folds/, no backfill


def main():
    t0 = time.time()

    core_daily_path = RAW_DIR / "forecasting_hackathon_core_daily.parquet"
    hourly_sales_path = EXTERNAL_DIR / "hourly_sales_daily.parquet"
    page_views_path = EXTERNAL_DIR / "page_views_daily.parquet"

    # Snapshot the input table, and (unless BACKFILL_FOLDS is off) backfill any
    # residual-stack folds this valid_date needs that earlier runs didn't leave in
    # folds/. In production folds accumulate across runs; a cold competition date
    # (unknown ahead of time) backfills them here, before the forecast.
    if BACKFILL_FOLDS:
        regen_missing_folds(VALID_DATE, seqgru_seeds=3, num_steps=8000, input_table=INPUT_TABLE)
    else:
        ensure_data(INPUT_TABLE)

    members = generate_members(
        core_daily_path, VALID_DATE, hourly_sales_path, page_views_path,
        seqgru_seeds=3, num_steps=8000,
    )

    submission = run_pipeline(VALID_DATE, members)
    # residual-stack correction trained on the folds/ (backfilled above)
    submission = apply_residual_stack(submission, members, VALID_DATE, core_daily_path)
    submission.to_csv(OUTPUT_CSV, index=False)
    elapsed_minutes = (time.time() - t0) / 60
    print(f"\ndone in {elapsed_minutes:.1f} min -> {OUTPUT_CSV}  ({len(submission):,} rows)")


if __name__ == "__main__":
    main()
