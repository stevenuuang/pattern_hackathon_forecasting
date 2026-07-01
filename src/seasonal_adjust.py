"""Christmas / New Year week shape correction from history only: scale each
event week by the prior-year analog's weekly index over the forecast's implied
index. Only fires when the analog week shares the event and weekday class
(mid-week Tue-Fri dips to ~0.85x on the shipping cutoff; Sat-Mon does not).
"""

from pathlib import Path

import pandas as pd

from .core_target import CORE_DAILY_PATH, load_core_instock

MIDWEEK = {1, 2, 3, 4}  # Tue-Fri in pandas dayofweek
FACTOR_CLIP = (0.75, 1.25)
MIDWEEK_XMAS_IDX = 0.84


def _thanksgiving(year: int) -> pd.Timestamp:
    nov1 = pd.Timestamp(year=year, month=11, day=1)
    return nov1 + pd.Timedelta(days=(3 - nov1.dayofweek) % 7 + 21)


def _events(year: int) -> dict[str, pd.Timestamp]:
    tg = _thanksgiving(year)
    return {
        "black_fri": tg + pd.Timedelta(days=1),
        "cyber_mon": tg + pd.Timedelta(days=4),
        "christmas": pd.Timestamp(year=year, month=12, day=25),
        "ny_day": pd.Timestamp(year=year + 1, month=1, day=1),
    }


def _week_events(week_start: pd.Timestamp) -> dict[str, pd.Timestamp]:
    """Events falling within [week_start, week_start + 6d]."""
    out = {}
    for year in {week_start.year - 1, week_start.year}:
        for name, date in _events(year).items():
            if week_start <= date <= week_start + pd.Timedelta(days=6):
                out[name] = date
    return out


def _analog_applies(week_start: pd.Timestamp) -> bool:
    """True if this is a Christmas/NY week whose -364d analog week contains
    the same event with the same weekday class."""
    events = _week_events(week_start)
    analog_events = _week_events(week_start - pd.Timedelta(days=364))
    for name in ("christmas", "ny_day"):
        if name in events:
            return name in analog_events and (
                (events[name].dayofweek in MIDWEEK)
                == (analog_events[name].dayofweek in MIDWEEK)
            )
    return False


def _midweek_xmas_fallback(week_start: pd.Timestamp) -> bool:
    """True if this week has a midweek (dip-class) Christmas but no weekday-
    class-matched -364d analog, so the analog correction would be skipped --
    fall back to the empirical midweek dip."""
    ev = _week_events(week_start)
    if "christmas" not in ev or ev["christmas"].dayofweek not in MIDWEEK:
        return False
    return not _analog_applies(week_start)


def seasonal_factors(
    forecast: pd.DataFrame, valid_date: str, core_path: Path = CORE_DAILY_PATH
) -> pd.Series:
    valid = pd.Timestamp(valid_date)
    horizon_weeks = int(forecast["horizon_week"].max())
    week_starts = {
        h: valid + pd.Timedelta(weeks=h - 1) for h in range(1, horizon_weeks + 1)
    }
    event_hs = [h for h, ws in week_starts.items() if _analog_applies(ws)]
    fb_hs = [h for h, ws in week_starts.items() if _midweek_xmas_fallback(ws)]
    factors = pd.Series(1.0, index=range(1, horizon_weeks + 1))
    if not event_hs and not fb_hs:
        return factors

    hist = load_core_instock(core_path, valid_date)
    offset = (hist["date"].dt.dayofweek + 1) % 7
    hist["week"] = hist["date"] - pd.to_timedelta(offset, unit="D")
    weekly = hist.groupby("week")["units"].sum()

    ref_hs = [h for h, ws in week_starts.items() if not _week_events(ws)]
    analog = {h: week_starts[h] - pd.Timedelta(days=364) for h in week_starts}
    hist_ref = weekly.reindex([analog[h] for h in ref_hs]).mean()
    fcst_totals = forecast.groupby("horizon_week")["forecast"].sum()
    fcst_ref = fcst_totals[ref_hs].mean()

    for h in event_hs:
        hist_val = weekly.get(analog[h])
        if pd.isna(hist_val) or hist_ref == 0 or fcst_ref == 0:
            continue
        factor = (hist_val / hist_ref) / (fcst_totals[h] / fcst_ref)
        factors[h] = min(max(factor, FACTOR_CLIP[0]), FACTOR_CLIP[1])

    for h in fb_hs:
        if fcst_ref == 0 or fcst_totals[h] == 0:
            continue
        factor = MIDWEEK_XMAS_IDX / (fcst_totals[h] / fcst_ref)
        factors[h] = min(max(factor, FACTOR_CLIP[0]), FACTOR_CLIP[1])
    return factors
