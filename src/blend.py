"""Weighted blend of member forecasts on the product + horizon keys.

Where a member is missing a row (some wrappers skip cold-start products) the
remaining weights are renormalized over the members present for that row.
"""
import numpy as np
import pandas as pd

from .core_target import PRODUCT_COLS

KEYS = PRODUCT_COLS + ["horizon_week", "horizon_date"]


def blend(inputs: list[tuple[pd.DataFrame, float]]) -> pd.DataFrame:
    merged, weights = None, []
    for i, (df, weight) in enumerate(inputs):
        df = df[KEYS + ["forecast"]].rename(columns={"forecast": f"f{i}"})
        weights.append(weight)
        merged = df if merged is None else merged.merge(df, on=KEYS, how="outer")
    vals = merged[[f"f{i}" for i in range(len(inputs))]].to_numpy()
    eff_w = ~np.isnan(vals) * np.array(weights)
    merged["forecast"] = np.nansum(vals * eff_w, axis=1) / eff_w.sum(axis=1)
    return merged[KEYS + ["forecast"]]
