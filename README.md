# Forecasting Hackathon — Deliverable

13-week weekly unit-sales forecast. Ensemble of fine-tuned Chronos-2 + TimesFM-2.5 + LightGBM, post-processed (seasonal/promo correction, level uplift, hierarchical
reconciliation to partner×marketplace aggregates), then a residual-stack
correction trained on backtest folds. Output: `submission.csv` (not a Snowflake table.)

## Run

```bash
uv sync
uv run python main.py
```

Set the inputs at the top of `main.py`:

```python
INPUT_TABLE    = "forecasting_hackathon_core_daily"  # input table name
VALID_DATE     = "2025-10-26"                         # a Sunday; first forecast week
BACKFILL_FOLDS = True                                 # see "Folds" below
```

Needs read-only Snowflake access (to snapshot the table) and ideally a CUDA GPU.
Much slower on MPS/CPU — the GRU and the daily Chronos fine-tune are the
bottlenecks; see the MPS runtime notes in `src/members.py`.

## Folds (residual stack)

The residual-stack correction trains on `folds/` — one wide matrix per historical
origin (member forecasts + blend + trailing volume), each built using only data
before that origin. `folds/` ships with 13 monthly origins ending 2025-10-26.

Because the actual `VALID_DATE` is unknown ahead of time, the folds it needs
can't all be pre-shipped. With `BACKFILL_FOLDS = True` (default), `main.py` calls
`regen_missing_folds()` first: it steps forward monthly from the newest existing
fold up to `VALID_DATE` and builds any origin that isn't in `folds/` yet (~2.5 h
each on a 5080). It is idempotent — a no-op when `folds/` is already current, so
in production, where folds accumulate across runs, the run is just the ~2.5 h
forecast. Set `BACKFILL_FOLDS = False` to skip backfilling and train the stack on
whatever folds are already present. This reduces runtime but may lead to lower performance.

To pre-build folds offline instead (before knowing/using the date):

```bash
uv run python regen_folds.py --valid-date yyyy-mm-dd   # only the missing origins
```

Generated files stay inside this folder: `data/`, `models/`, `folds/`, and
`submission.csv`.
