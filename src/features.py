"""Feature engineering for glucose prediction.

For non-ML readers: see docs/ML_PRIMER.md first. The short version:
the XGBoost model never sees the raw time-series. It sees a *table* of
"snapshot features" — one row per 5-minute step — where each column
captures something useful about what's happening right now or what
just happened. This file is where we build that table.

Design rules:
1. Every feature must be computable from data **available at time t**.
   No peeking into the future. This is the #1 source of leakage in
   time-series ML.
2. Every feature has a comment explaining (a) what it is, (b) why it
   helps glucose prediction. Read the comments — they're for you.
3. The target ("label") is glucose `HORIZON_MIN` minutes in the
   future. We compute it once and align it to each row's timestamp.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# How far ahead we predict. 30 minutes is the standard short-horizon
# glucose-prediction benchmark — useful enough to act on, close enough
# to be reliably forecastable.
HORIZON_MIN = 30
HORIZON_STEPS = HORIZON_MIN // 5  # 5-minute grid -> 6 steps

# Stretch target for comparison; produced but not the primary label.
HORIZON_LONG_MIN = 60
HORIZON_LONG_STEPS = HORIZON_LONG_MIN // 5

# Lag offsets in minutes: glucose value N minutes ago.
# Why these specifically:
#   5/10/15/30 = short-term momentum (the model leans heavily here)
#   60/120     = medium-term context ("was there a recent meal spike?")
#   240        = ~4hr — captures slow drift / dawn phenomenon onset
LAG_MINUTES = (5, 10, 15, 30, 60, 120, 240)

# Rolling-window sizes in minutes for mean/std/slope calculations.
ROLLING_WINDOWS_MIN = (15, 30, 60, 120)

# How much past insulin we sum into "recent bolus" features.
BOLUS_WINDOWS_MIN = (15, 30, 60, 120)


def _ensure_grid(df: pd.DataFrame) -> pd.DataFrame:
    """Sanity check: the merged timeline must already be on the 5-min grid."""
    df = df.sort_values("timestamp").copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    diffs = df["timestamp"].diff().dropna().dt.total_seconds() / 60.0
    if not (diffs == 5).all():
        # Don't reject — the merge_pipeline should have handled it,
        # but warn loudly so debugging starts in the right place.
        logger.warning("Timeline is not on a strict 5-min grid (max gap = %s min). Re-run merge_pipeline.", diffs.max())
    return df


def _add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Glucose value N minutes ago.

    These are the strongest single predictors of short-horizon glucose.
    Implementation: pandas .shift(steps) where steps = minutes / 5.
    """
    for m in LAG_MINUTES:
        steps = m // 5
        df[f"glucose_lag_{m}m"] = df["glucose_mg_dl"].shift(steps)

    # First difference: how much has glucose moved in the last 5/15/30 min?
    # This is a hand-rolled "velocity" feature; cheap and very informative.
    df["glucose_delta_5m"] = df["glucose_mg_dl"] - df["glucose_mg_dl"].shift(1)
    df["glucose_delta_15m"] = df["glucose_mg_dl"] - df["glucose_mg_dl"].shift(3)
    df["glucose_delta_30m"] = df["glucose_mg_dl"] - df["glucose_mg_dl"].shift(6)
    return df


def _add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """Mean, std-dev, and slope of glucose over recent windows.

    The window is *closed on the left* and excludes the current row's
    "future" — pandas rolling() with min_periods handles this correctly
    as long as the column is the past data. We do NOT include t+1 etc.
    """
    glucose = df["glucose_mg_dl"]
    for m in ROLLING_WINDOWS_MIN:
        window = m // 5
        roll = glucose.rolling(window, min_periods=max(2, window // 2))
        df[f"glucose_mean_{m}m"] = roll.mean()
        # Std-dev = volatility. High std = unstable glucose, harder to
        # predict; the model can learn to widen its margins or use
        # different rules in volatile vs stable regimes.
        df[f"glucose_std_{m}m"] = roll.std()
        # Slope via simple linear regression on the window. We
        # vectorize a rolling apply() because numpy.polyfit per-window
        # is slow; the closed-form slope of (x, y) where x is evenly
        # spaced minutes is just (y_last - y_first) / (n - 1) bins.
        # That's not a *least squares* slope, but it's a fast,
        # noise-robust enough proxy for short windows.
        df[f"glucose_slope_{m}m"] = (glucose - glucose.shift(window - 1)) / max(window - 1, 1)
    return df


def _add_insulin_features(df: pd.DataFrame) -> pd.DataFrame:
    """Insulin-context features.

    IOB itself is already in the input. We add:
      - Sum of bolus units in recent windows ("did the user just eat?")
      - Time since last bolus (long times -> baseline state)
      - Boolean current-state flags (already in unified parquet)
    """
    for m in BOLUS_WINDOWS_MIN:
        window = m // 5
        # rolling().sum() over the bolus_units column: how many U were
        # delivered in the trailing window. A spike here predicts a
        # spike-then-drop in glucose ~30-90 min later (carb absorption
        # + insulin onset don't perfectly cancel).
        df[f"bolus_sum_{m}m"] = df["bolus_units"].fillna(0).rolling(window, min_periods=1).sum()

    # Time since last bolus, in minutes. We compute this by forward-
    # filling the timestamp at every nonzero-bolus row and subtracting.
    bolus_times = df["timestamp"].where(df["bolus_units"].fillna(0) > 0).ffill()
    df["minutes_since_bolus"] = ((df["timestamp"] - bolus_times).dt.total_seconds() / 60.0).fillna(9999.0)

    # Cap to avoid the model treating "10000 minutes" as meaningfully
    # different from "1000 minutes" — anything > 8h is effectively
    # "no recent bolus".
    df["minutes_since_bolus"] = df["minutes_since_bolus"].clip(upper=480.0)

    # Suspended / CIQ-active are already booleans; just make sure
    # they're 0/1 ints for XGBoost.
    df["is_suspended"] = df["is_suspended"].astype("int8")
    df["control_iq_active"] = df["control_iq_active"].astype("int8")
    return df


def _add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Circadian and weekday features.

    Why sin/cos: hour-of-day is *cyclic* — 23 and 0 should be close to
    each other, but as raw integers they're maximally distant. Encoding
    as (sin(2*pi*h/24), cos(2*pi*h/24)) puts midnight neighbors at
    nearly identical coordinates, which a tree can split on cleanly.

    Glucose has strong circadian patterns — dawn phenomenon (3-7am
    cortisol rise), evening insulin sensitivity drop, etc — so this
    block routinely shows up in the top-10 feature importances.
    """
    ts = df["timestamp"].dt
    hour = ts.hour + ts.minute / 60.0
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    df["dow"] = ts.dayofweek.astype("int8")  # 0=Mon, 6=Sun
    df["is_weekend"] = (df["dow"] >= 5).astype("int8")
    return df


def _add_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Compute the labels we'll train against.

    `target_30m` = glucose value 30 minutes after this row's timestamp.
    We use .shift(-N) to look ahead, then drop rows where the target
    is NaN (the tail of the series can't see its own future).
    """
    df["target_30m"] = df["glucose_mg_dl"].shift(-HORIZON_STEPS)
    df["target_60m"] = df["glucose_mg_dl"].shift(-HORIZON_LONG_STEPS)
    return df


# Columns we drop before training because they would leak the future,
# duplicate the target, or aren't features. `glucose_mg_dl` itself IS a
# feature (current value) — we keep it. The leaks come from `target_*`.
NON_FEATURE_COLS = ("timestamp", "target_30m", "target_60m")


def build_features(unified: pd.DataFrame) -> pd.DataFrame:
    """Top-level: take the unified parquet -> a feature table.

    Returns a dataframe with timestamp + features + targets. The caller
    (training script) is responsible for splitting and dropping rows
    that have any NaN target.
    """
    df = _ensure_grid(unified)
    df = _add_lag_features(df)
    df = _add_rolling_features(df)
    df = _add_insulin_features(df)
    df = _add_time_features(df)
    df = _add_targets(df)

    # Cast float columns to float32 — halves memory and XGBoost is
    # happier with it.
    for col in df.columns:
        if df[col].dtype == "float64":
            df[col] = df[col].astype("float32")

    return df


def feature_columns(df: pd.DataFrame) -> list[str]:
    """The columns the model should actually train on."""
    return [c for c in df.columns if c not in NON_FEATURE_COLS]


def run(input_path: Path, output_path: Path) -> pd.DataFrame:
    logger.info("Loading unified timeline: %s", input_path)
    unified = pd.read_parquet(input_path)
    feats = build_features(unified)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    feats.to_parquet(output_path, index=False)
    logger.info(
        "Wrote %d rows x %d cols to %s (target_30m non-null: %d)",
        len(feats), len(feats.columns), output_path, feats["target_30m"].notna().sum(),
    )
    return feats


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build feature table from unified timeline.")
    p.add_argument("--input", type=Path, default=Path("data/processed/unified_timeline.parquet"))
    p.add_argument("-o", "--output", type=Path, default=Path("data/processed/features.parquet"))
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    if not args.input.exists():
        logger.error("Missing unified parquet: %s", args.input)
        return 2
    run(args.input, args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
