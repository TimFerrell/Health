"""Drift / live-accuracy tracking.

Each refresh cycle, refresh.py appends a row to predictions_log.parquet:
    predicted_at         — when we made the prediction
    target_timestamp     — predicted_at + 30 min (when truth will arrive)
    current_glucose      — glucose at predicted_at
    prediction_30m       — what we forecast for target_timestamp

This module joins that log with the unified timeline (which holds the
*actual* future glucose), computes per-prediction error, and exposes
rolling MAE/RMSE so we can see whether the model is silently degrading.

When does drift happen for T1D?
  * Sensor change (G7's first 12-24h have different bias).
  * Growth spurt or insulin-sensitivity shift (kids: every few weeks).
  * Schedule change (school year vs summer break).
  * Pump site rotation issues.

When the rolling MAE crosses ~1.5x the trained-test MAE, you should
retrain. We surface a recommendation — we don't auto-retrain (yet),
because retraining on drifted data without inspection can entrench
sensor bugs.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


PREDICTION_LOG_SCHEMA = [
    "predicted_at",
    "target_timestamp",
    "current_glucose",
    "prediction_30m",
]


@dataclass
class DriftStatus:
    n_scored: int
    mae_24h: float
    rmse_24h: float
    mae_7d: float
    rmse_7d: float
    mae_all: float
    trained_mae: float | None
    drift_ratio: float | None        # rolling 24h MAE / trained MAE
    recommend_retrain: bool
    recommendation: str


def append_log(log_path: Path, row: dict) -> None:
    """Append one prediction record to the log parquet (creates if absent)."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    new = pd.DataFrame([row])
    new["predicted_at"] = pd.to_datetime(new["predicted_at"], utc=True)
    new["target_timestamp"] = pd.to_datetime(new["target_timestamp"], utc=True)
    if log_path.exists():
        existing = pd.read_parquet(log_path)
        combined = pd.concat([existing, new], ignore_index=True)
    else:
        combined = new
    combined = combined.drop_duplicates(subset=["predicted_at"], keep="last").sort_values("predicted_at")
    combined.to_parquet(log_path, index=False)


def join_with_actuals(log: pd.DataFrame, unified: pd.DataFrame) -> pd.DataFrame:
    """Left-join the log with unified[timestamp, glucose] on target_timestamp."""
    if log.empty:
        return log.assign(actual_30m=pd.Series(dtype="float32"), abs_error=pd.Series(dtype="float32"))

    truth = unified[["timestamp", "glucose_mg_dl"]].rename(columns={
        "timestamp": "target_timestamp",
        "glucose_mg_dl": "actual_30m",
    })
    truth["target_timestamp"] = pd.to_datetime(truth["target_timestamp"], utc=True)
    log = log.copy()
    log["target_timestamp"] = pd.to_datetime(log["target_timestamp"], utc=True)
    merged = log.merge(truth, on="target_timestamp", how="left")
    merged["abs_error"] = (merged["prediction_30m"] - merged["actual_30m"]).abs()
    return merged


def _window_metrics(scored: pd.DataFrame, since: pd.Timestamp) -> tuple[float, float]:
    sub = scored[scored["target_timestamp"] >= since].dropna(subset=["actual_30m"])
    if sub.empty:
        return float("nan"), float("nan")
    err = sub["prediction_30m"] - sub["actual_30m"]
    return float(err.abs().mean()), float(np.sqrt((err ** 2).mean()))


def compute_status(
    log_path: Path,
    unified_path: Path,
    metrics_path: Path | None,
    drift_ratio_threshold: float = 1.5,
) -> DriftStatus:
    if not log_path.exists():
        return DriftStatus(0, *([float("nan")] * 5), None, None, False,
                           "No prediction log yet — run refresh at least twice.")
    log = pd.read_parquet(log_path)
    unified = pd.read_parquet(unified_path) if unified_path.exists() else pd.DataFrame(
        columns=["timestamp", "glucose_mg_dl"]
    )
    scored = join_with_actuals(log, unified)
    scored_only = scored.dropna(subset=["actual_30m"])

    now = pd.Timestamp.now(tz="UTC")
    mae_24h, rmse_24h = _window_metrics(scored, now - pd.Timedelta("24h"))
    mae_7d, rmse_7d = _window_metrics(scored, now - pd.Timedelta("7d"))
    mae_all = float(scored_only["abs_error"].mean()) if not scored_only.empty else float("nan")

    trained_mae: float | None = None
    if metrics_path and metrics_path.exists():
        try:
            trained_mae = float(json.loads(metrics_path.read_text())["mae"])
        except (KeyError, ValueError):
            trained_mae = None

    drift_ratio = None
    recommend = False
    if trained_mae and not np.isnan(mae_24h):
        drift_ratio = mae_24h / trained_mae
        recommend = drift_ratio >= drift_ratio_threshold

    if recommend:
        msg = (f"Live 24h MAE ({mae_24h:.1f}) is {drift_ratio:.2f}× the trained-test MAE "
               f"({trained_mae:.1f}). Consider retraining.")
    elif trained_mae and not np.isnan(mae_24h):
        msg = (f"Live accuracy in line with training "
               f"(24h MAE {mae_24h:.1f} vs trained {trained_mae:.1f}, ratio {drift_ratio:.2f}).")
    else:
        msg = f"Scored {len(scored_only)} predictions. Need a trained model + more time to assess drift."

    return DriftStatus(
        n_scored=len(scored_only),
        mae_24h=mae_24h, rmse_24h=rmse_24h,
        mae_7d=mae_7d, rmse_7d=rmse_7d,
        mae_all=mae_all,
        trained_mae=trained_mae,
        drift_ratio=drift_ratio,
        recommend_retrain=recommend,
        recommendation=msg,
    )


def rolling_mae_series(log_path: Path, unified_path: Path, window: str = "6h") -> pd.DataFrame:
    """Return a time-indexed series of rolling MAE for plotting."""
    if not log_path.exists() or not unified_path.exists():
        return pd.DataFrame(columns=["target_timestamp", "rolling_mae"])
    scored = join_with_actuals(pd.read_parquet(log_path), pd.read_parquet(unified_path))
    scored = scored.dropna(subset=["actual_30m"]).sort_values("target_timestamp")
    if scored.empty:
        return pd.DataFrame(columns=["target_timestamp", "rolling_mae"])
    scored = scored.set_index("target_timestamp")
    rolled = scored["abs_error"].rolling(window).mean().reset_index()
    rolled = rolled.rename(columns={"abs_error": "rolling_mae"})
    return rolled
