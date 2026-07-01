"""Member generation for one valid_date — the single source of truth shared by
the graded run (main.py) and the backtest regen (regen_folds.py).

Producing members here (rather than duplicating the orchestration) guarantees the
backtest folds are generated with EXACTLY the configs the graded run uses, so the
residual-stack training matrices and the deployed blend never drift.

MPS runtime notes (why a Mac run is far slower than the ~2.5 h 5080 run):
- seqgru is the biggest bottleneck: nn.GRU has no fused Metal kernel, so MPS unrolls
  the recurrence one timestep at a time (CUDA runs the whole sequence in a single
  cuDNN kernel) — 3 seeds x 25 epochs x 16 origins of that serial fallback dominates.
- The daily chronos finetune (the FIRST one) is slow: daily grain is ~7x longer sequences 
  than weekly, so the transformer is compute-bound and the
  5080's throughput edge is largest there (plus one-time MPS shader compilation).
- The weekly chronos finetune (the SECOND) runs about the same on MPS and 5080: short
  weekly sequences make it overhead-bound (kernel launch / host sync), not FLOP-bound,
  so the underutilized 5080 shows no real advantage.
"""
import gc
import subprocess
import sys
import tempfile

import pandas as pd
import torch

from .build_series import build_series
from .build_agg_series import build_agg_series, build_agg_series_daily
from .models import chronos2, seqgru, timesfm25


def _free():
    """Release the MPS caching allocator + dead Python objects between model
    stages (two 8000-step chronos fits otherwise SIGSEGV the next load on MPS).
    No-op beyond a GC pass on cuda/cpu."""
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def _gbm_member(valid_date, log_target=False):
    """Train a LightGBM member in a torch-free subprocess (LightGBM + torch each
    bundle OpenMP and segfault sharing a process on macOS). Reads the fixed
    CORE_DAILY_PATH snapshot (same for every fold)."""
    with tempfile.NamedTemporaryFile(suffix=".parquet") as f:
        cmd = [sys.executable, "-m", "src.models.gbm", "--valid-date", valid_date,
               "-o", f.name]
        if log_target:
            cmd.append("--log-target")
        subprocess.run(cmd, check=True)
        return pd.read_parquet(f.name)


def generate_members(core_daily_path, valid_date, hourly_sales_path, page_views_path,
                     seqgru_seeds=3, num_steps=8000) -> dict:
    """Generate every pipeline member for one valid_date, returned as a dict keyed
    by the pipeline member name. Identical orchestration to the graded main.py."""
    weekly_uncensored = build_series(core_daily_path, valid_date, "W", uncensor=True)
    daily_series = build_series(core_daily_path, valid_date, "D",
                                partner_path=hourly_sales_path,
                                pageviews_path=page_views_path)

    chronos_daily_ckpt = chronos2.finetune(daily_series, valid_date, freq="D", num_steps=num_steps)
    _free()
    chronos_weekly_ckpt = chronos2.finetune(weekly_uncensored, valid_date, freq="W", num_steps=num_steps)
    _free()
    timesfm_weekly_ckpt = timesfm25.finetune(weekly_uncensored, valid_date, freq="W",
                                             num_steps=num_steps, learning_rate=1e-4)
    _free()

    members = {}
    members["chronos2_ft8000pcov_pv_D_q0.55"] = chronos2.predict(
        daily_series, valid_date, freq="D", quantile=0.55, model_path=str(chronos_daily_ckpt))
    members["chronos2_ft8000pcov_pv_D_q0.5"] = chronos2.predict(
        daily_series, valid_date, freq="D", quantile=0.5, model_path=str(chronos_daily_ckpt))
    _free()
    members["chronos2_ft8000unc_W_q0.55"] = chronos2.predict(
        weekly_uncensored, valid_date, freq="W", quantile=0.55, model_path=str(chronos_weekly_ckpt))
    members["chronos2_ft8000unc_W_q0.6"] = chronos2.predict(
        weekly_uncensored, valid_date, freq="W", quantile=0.6, model_path=str(chronos_weekly_ckpt))
    _free()
    members["timesfm25_ft8000unc_W_q0.6"] = timesfm25.predict(
        weekly_uncensored, valid_date, freq="W", quantile=0.6, model_path=str(timesfm_weekly_ckpt))
    _free()
    members["timesfm25_zeroshot_W_unc_q0.7"] = timesfm25.predict(
        weekly_uncensored, valid_date, freq="W", quantile=0.7)
    _free()
    members["timesfm25_zeroshot_D_q0.6"] = timesfm25.predict(
        daily_series, valid_date, freq="D", quantile=0.6)
    _free()
    members["gbm_cat"] = _gbm_member(valid_date)
    members["gbm_log"] = _gbm_member(valid_date, log_target=True)
    members["seqgru_W_q0.55"] = seqgru.ensemble_predict(
        valid_date, weekly_uncensored, seeds=seqgru_seeds, quantile=0.55, epochs=25,
        hid=192, layers=2, dropout=0.1, weight_decay=1e-5, origins=16, cosine=True)
    _free()
    members["reconcile_agg"] = chronos2.predict(
        build_agg_series(core_daily_path, valid_date), valid_date, freq="W", quantile=0.6)
    members["reconcile_agg_D"] = chronos2.predict(
        build_agg_series_daily(daily_series), valid_date, freq="D", quantile=0.6)
    _free()
    return members
