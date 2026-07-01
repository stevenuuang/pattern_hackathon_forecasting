"""TimesFM 2.5 (google/timesfm-2.5-200m-pytorch) wrapper: zero-shot predict +
LoRA fine-tune. Target-only; NaN periods (no in-stock days) are filled with 0.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ..device import get_device
from .common import KEYS, to_submission

MODEL_ID = "google/timesfm-2.5-200m-pytorch"  # timesfm library checkpoint (zero-shot predict)
HF_MODEL_ID = "google/timesfm-2.5-200m-transformers"  # HF Transformers port (fine-tune + ft inference)


def _hf_quantile_forecast(
    inputs: list[np.ndarray], steps: int, batch_size: int, device: str | None,
    model_path: str,
) -> np.ndarray:
    """(n, steps, 10) quantile forecasts from the HF checkpoint + a LoRA
    adapter saved by finetune(). The fine-tune path trains the HF port (layout
    differs from the timesfm-library checkpoint), so fine-tuned inference goes
    through HF too. Needs transformers >= 5, like finetune()."""
    from peft import PeftModel
    from transformers import TimesFm2_5ModelForPrediction

    dev = device or get_device()
    dtype = torch.bfloat16 if dev == "cuda" else torch.float32
    model = TimesFm2_5ModelForPrediction.from_pretrained(
        HF_MODEL_ID, dtype=dtype, device_map=dev
    )
    patch, ctx_max = model.config.patch_length, model.config.context_length
    model = PeftModel.from_pretrained(model, model_path)
    model.eval()
    # Size the forecast context to the actual data (rounded to a patch). Left
    # unset, the HF forward pads every series to context_length (16384), which
    # dominates MPS runtime; masked padding means the forecast is unchanged but
    # this is ~40x faster.
    ctx_len = min(ctx_max, ((max(len(s) for s in inputs) + patch - 1) // patch) * patch)
    chunks = []
    with torch.inference_mode():
        for i in range(0, len(inputs), batch_size):
            batch = [torch.from_numpy(s).to(dev, dtype) for s in inputs[i : i + batch_size]]
            out = model(past_values=batch, truncate_negative=True, forecast_context_len=ctx_len)
            chunks.append(out.full_predictions[:, :steps, :].float().cpu().numpy())
    return np.concatenate(chunks)


def _assemble(keys, forecast, valid_date, steps, freq):
    """Tile per-product forecasts over the horizon into the submission schema."""
    valid = pd.Timestamp(valid_date)
    step_days = 7 if freq == "W" else 1
    dates = pd.to_datetime([valid + pd.Timedelta(days=step_days * i) for i in range(steps)])
    out = keys.loc[keys.index.repeat(steps)].reset_index(drop=True).assign(
        horizon_date=np.tile(dates, len(keys)),
        forecast=forecast.reshape(-1),
    )
    return to_submission(out, valid_date)


def predict(
    series_df: pd.DataFrame,
    valid_date: str,
    horizon_weeks: int = 13,
    freq: str = "W",
    quantile: float = 0.5,
    batch_size: int = 128,
    device: str | None = None,
    model_path: str | None = None,
) -> pd.DataFrame:
    """Pass a LoRA adapter directory from finetune() as model_path for
    fine-tuned inference (runs via the HF checkpoint)."""
    import timesfm

    period = "week_start" if freq == "W" else "date"
    steps = horizon_weeks if freq == "W" else horizon_weeks * 7
    valid = pd.Timestamp(valid_date)
    max_context = 2048 if freq == "D" else 512

    context = series_df[series_df[period] < valid].sort_values(period)
    grouped = context.groupby(KEYS)["target"]
    keys = pd.DataFrame(grouped.groups.keys(), columns=KEYS)
    inputs = [s.fillna(0.0).to_numpy(dtype=np.float32) for _, s in grouped]

    if model_path is not None:
        # same context budget as the library path; quantile layout matches
        # (index 0 = point, 1..9 = q0.1..q0.9)
        inputs = [s[-max_context:] for s in inputs]
        quantiles = _hf_quantile_forecast(inputs, steps, batch_size, device, model_path)
        forecast = quantiles[:, :steps, int(round(quantile * 10))]
        return _assemble(keys, forecast, valid_date, steps, freq)

    model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(MODEL_ID)
    # the library only auto-selects cuda/cpu and reads self.device for input
    # placement, so set both the attribute and the weights (enables mps)
    dev = torch.device(device or get_device())
    model.model.device = dev
    model.model.to(dev)
    model.compile(
        timesfm.ForecastConfig(
            max_context=max_context,
            max_horizon=steps,
            normalize_inputs=True,
            use_continuous_quantile_head=True,
            infer_is_positive=True,
            fix_quantile_crossing=True,
            # internal decode batch; the default of 1 forecasts one series at
            # a time and leaves the device idle
            per_core_batch_size=batch_size,
        )
    )

    # quantile output is (batch, horizon, 10): index 0 = point, 1..9 = q0.1..q0.9
    q_index = int(round(quantile * 10))
    # the library fills partial batches with float64 dummy series, which mps
    # rejects; pad to a full batch with float32 dummies and drop them after
    n = len(inputs)
    inputs += [np.zeros(3, dtype=np.float32)] * (-n % batch_size)
    _, quantiles = model.forecast(horizon=steps, inputs=inputs)
    forecast = quantiles[:n, :steps, q_index]
    return _assemble(keys, forecast, valid_date, steps, freq)


def finetune(
    series_df: pd.DataFrame,
    valid_date: str,
    horizon_weeks: int = 13,
    freq: str = "W",
    num_steps: int = 1000,
    batch_size: int = 32,
    learning_rate: float = 1e-4,
    context_len: int | None = None,
    lora_r: int = 4,
    lora_alpha: int = 8,
    output_dir: Path | str | None = None,
    device: str | None = None,
    seed: int = 0,
) -> Path:
    """LoRA fine-tuning of the HF Transformers checkpoint, following the
    official example (timesfm-forecasting/examples/finetuning/finetune_lora.py).
    Random (context, horizon) windows from history before valid_date; the HF
    model computes the loss. Returns the saved adapter directory.

    Needs transformers >= 5 (TimesFm2_5ModelForPrediction); the whole stack is
    pinned to transformers 5 (see pyproject.toml), so this runs in-process.
    """
    try:
        from transformers import TimesFm2_5ModelForPrediction
    except ImportError as err:
        raise ImportError(finetune.__doc__) from err
    from peft import LoraConfig, get_peft_model

    period = "week_start" if freq == "W" else "date"
    steps = horizon_weeks if freq == "W" else horizon_weeks * 7
    if context_len is None:
        context_len = 64 if freq == "W" else 256  # must be a multiple of 32
    valid = pd.Timestamp(valid_date)

    context = series_df[series_df[period] < valid].sort_values(period)
    series = [
        s.fillna(0.0).to_numpy(dtype=np.float32)
        for _, s in context.groupby(KEYS)["target"]
        if len(s) >= context_len + steps
    ]
    if not series:
        raise ValueError(f"no series with >= {context_len + steps} periods to fine-tune on")

    dev = device or get_device()
    # bf16 matmuls crash Metal kernels, so only use bf16 on cuda
    dtype = torch.bfloat16 if dev == "cuda" else torch.float32
    model = TimesFm2_5ModelForPrediction.from_pretrained(HF_MODEL_ID, dtype=dtype, device_map=dev)
    model = get_peft_model(
        model,
        LoraConfig(r=lora_r, lora_alpha=lora_alpha, target_modules="all-linear", bias="none"),
    )

    rng = np.random.default_rng(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    model.train()
    for step in range(num_steps):
        rows = [series[i] for i in rng.integers(0, len(series), batch_size)]
        starts = [rng.integers(0, len(s) - context_len - steps + 1) for s in rows]
        past = np.stack([s[i : i + context_len] for s, i in zip(rows, starts)])
        future = np.stack(
            [s[i + context_len : i + context_len + steps] for s, i in zip(rows, starts)]
        )
        out = model(
            past_values=torch.from_numpy(past).to(dev, dtype=dtype),
            future_values=torch.from_numpy(future).to(dev, dtype=dtype),
            forecast_context_len=context_len,
        )
        out.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()
        if step % 50 == 0 or step == num_steps - 1:
            print(f"step {step}: loss {out.loss.item():.4f}")

    output_dir = Path(output_dir or f"models/timesfm25_{freq}_{valid_date}")
    model.save_pretrained(output_dir)
    return output_dir
