"""Chronos-2 (amazon/chronos-2) wrapper: zero-shot predict + LoRA fine-tune.

Event flags + promo are known-future covariates; partner_units / page_views are
past-only covariates. Series with < 3 context points get a flat mean fallback.
"""

from pathlib import Path

import pandas as pd

from ..build_series import EVENT_COLS
from ..device import get_device
from .common import pack_id, to_submission, unpack_id

MODEL_ID = "amazon/chronos-2"
COVARIATES = ["promo_pct_off"] + EVENT_COLS
PAST_COVARIATES = ["partner_units", "page_views"]  # context-only; never in future_df


def _frames(
    series_df: pd.DataFrame, valid_date: str, horizon_weeks: int, freq: str
) -> tuple[pd.DataFrame, pd.DataFrame, str, int]:
    """Context and future-covariate frames in predict_df format."""
    period = "week_start" if freq == "W" else "date"
    steps = horizon_weeks if freq == "W" else horizon_weeks * 7
    valid = pd.Timestamp(valid_date)

    df = series_df.assign(
        item_id=pack_id,
        promo_pct_off=lambda d: d["promo_pct_off"].fillna(0.0),
        **series_df[EVENT_COLS].fillna(0).astype(float),
    )
    past_covs = [c for c in PAST_COVARIATES if c in df.columns]
    cols = ["item_id", period, "target"] + COVARIATES + past_covs
    context = df.loc[df[period] < valid, cols]
    future = df.loc[
        (df[period] >= valid) & (df[period] < valid + pd.Timedelta(days=7 * horizon_weeks)),
        [c for c in cols if c != "target" and c not in past_covs],
    ]
    return context, future, period, steps


def predict(
    series_df: pd.DataFrame,
    valid_date: str,
    horizon_weeks: int = 13,
    freq: str = "W",
    quantile: float = 0.5,
    batch_size: int = 256,
    device: str | None = None,
    model_path: str = MODEL_ID,
) -> pd.DataFrame:
    """Forecast in the submission schema. Pass a fine-tuned checkpoint
    directory as model_path to use a fine-tuned model."""
    from chronos import Chronos2Pipeline

    context, future, period, steps = _frames(series_df, valid_date, horizon_weeks, freq)
    dev = device or get_device()

    # Daily predict needs a smaller batch on MPS to avoid memory errors.
    if freq == "D" and dev == "mps":
        batch_size = min(batch_size, 32)

    counts = context.groupby("item_id").size()
    model_ids = counts[counts >= 3].index
    short_ids = future["item_id"].unique()[~pd.Index(future["item_id"].unique()).isin(model_ids)]

    pipeline = Chronos2Pipeline.from_pretrained(model_path, device_map=dev)
    pred = pipeline.predict_df(
        context[context["item_id"].isin(model_ids)],
        future_df=future[future["item_id"].isin(model_ids)],
        prediction_length=steps,
        quantile_levels=[quantile],
        id_column="item_id",
        timestamp_column=period,
        target="target",
        batch_size=batch_size,
    )
    out = unpack_id(pred["item_id"]).assign(
        horizon_date=pd.to_datetime(pred[period]),
        forecast=pred[str(quantile)],
    )

    if len(short_ids):
        means = context.groupby("item_id")["target"].mean().reindex(short_ids).fillna(0.0)
        fallback = future.loc[future["item_id"].isin(short_ids), ["item_id", period]]
        fb = unpack_id(fallback["item_id"]).assign(
            horizon_date=pd.to_datetime(fallback[period]),
            forecast=means.reindex(fallback["item_id"]).to_numpy(),
        )
        out = pd.concat([out, fb], ignore_index=True)

    return to_submission(out, valid_date)


def finetune(
    series_df: pd.DataFrame,
    valid_date: str,
    horizon_weeks: int = 13,
    freq: str = "W",
    mode: str = "lora",
    num_steps: int = 1000,
    learning_rate: float = 1e-5,
    batch_size: int = 32,
    output_dir: Path | str | None = None,
    device: str | None = None,
    model_path: str = MODEL_ID,
) -> Path:
    """Fine-tune on history before valid_date via the native Chronos2Pipeline.fit
    (LoRA by default). Returns the checkpoint directory, usable as
    predict(model_path=...). Covariate columns are passed so the model learns
    which ones are known into the future."""
    from chronos import Chronos2Pipeline
    from chronos.df_utils import convert_df_input_to_list_of_dicts_input

    context, future, period, steps = _frames(series_df, valid_date, horizon_weeks, freq)
    # drop cold-start series the model rejects (<3 context points); predict()
    # covers them with the mean-history fallback
    counts = context.groupby("item_id").size()
    long_ids = counts[counts >= 3].index
    context = context[context["item_id"].isin(long_ids)]
    future = future[future["item_id"].isin(long_ids)]
    inputs, _, _ = convert_df_input_to_list_of_dicts_input(
        context,
        future,
        target_columns=["target"],
        prediction_length=steps,
        id_column="item_id",
        timestamp_column=period,
    )

    output_dir = Path(output_dir or f"models/chronos2_{freq}_{valid_date}")
    pipeline = Chronos2Pipeline.from_pretrained(model_path, device_map=device or get_device())
    pipeline.fit(
        inputs,
        prediction_length=steps,
        finetune_mode=mode,
        learning_rate=learning_rate,
        num_steps=num_steps,
        batch_size=batch_size,
        output_dir=output_dir,
    )
    return output_dir / "finetuned-ckpt"
