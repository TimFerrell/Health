"""Threshold-based anomaly / alert generator.

For non-ML readers: this module turns a forecast number into a small
list of plain-English alerts. There's no learning here — these are
hand-tuned rules of the form "if the model predicts < 70 in 30 min,
flag a probable low".

Rules implemented:
  * predicted_low      — model says glucose will be < 70 mg/dL.
  * predicted_high     — model says glucose will be > 250 mg/dL.
  * predicted_drop     — model says glucose will fall by > 50 mg/dL
                         in 30 min (rapid-drop warning even if the
                         absolute level isn't low yet).
  * predicted_rise     — model says glucose will rise by > 60 mg/dL
                         in 30 min (carb-spike warning).
  * stale_data         — most recent CGM reading is > 15 min old.
  * sensor_extreme     — current glucose is the Low/High sentinel
                         (39 or 401), so all predictions downstream
                         are unreliable.

Severity:
  info     — context, no action implied.
  warn     — pay attention.
  critical — act now (low / very high / sensor-out-of-range).

These thresholds are sensible defaults, not clinical. Tune to taste
in DEFAULT_THRESHOLDS or pass a custom dict to detect().
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from typing import Iterable

import pandas as pd

DEFAULT_THRESHOLDS = {
    "low_mg_dl": 70.0,
    "high_mg_dl": 250.0,
    "drop_mg_dl_30m": 50.0,
    "rise_mg_dl_30m": 60.0,
    "stale_minutes": 15.0,
    "low_sentinel": 39.0,
    "high_sentinel": 401.0,
}


@dataclass
class Alert:
    type: str
    severity: str  # info | warn | critical
    message: str
    detail: dict

    def to_dict(self) -> dict:
        return asdict(self)


def _as_utc(ts: object) -> pd.Timestamp:
    t = pd.Timestamp(ts)
    return t.tz_localize("UTC") if t.tz is None else t.tz_convert("UTC")


def detect(
    *,
    current_glucose: float,
    current_timestamp: object,
    prediction_30m: float,
    now: object | None = None,
    thresholds: dict | None = None,
) -> list[Alert]:
    """Return a list of alerts for the given reading + forecast.

    `now` defaults to "now in UTC" — passed in for testability.
    """
    th = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    now_ts = _as_utc(now) if now is not None else pd.Timestamp.now(tz="UTC")
    cur_ts = _as_utc(current_timestamp)
    delta_30m = float(prediction_30m) - float(current_glucose)
    age_minutes = (now_ts - cur_ts).total_seconds() / 60.0

    alerts: list[Alert] = []

    # -- Sensor-out-of-range first; it invalidates the rest. ------------
    if current_glucose <= th["low_sentinel"]:
        alerts.append(Alert(
            "sensor_extreme", "critical",
            f"Sensor reads LOW (≤{int(th['low_sentinel'])} mg/dL). Confirm with a fingerstick.",
            {"current_glucose": current_glucose},
        ))
    elif current_glucose >= th["high_sentinel"]:
        alerts.append(Alert(
            "sensor_extreme", "critical",
            f"Sensor reads HIGH (≥{int(th['high_sentinel'])} mg/dL). Confirm with a fingerstick.",
            {"current_glucose": current_glucose},
        ))

    # -- Stale CGM. -----------------------------------------------------
    if age_minutes > th["stale_minutes"]:
        alerts.append(Alert(
            "stale_data", "warn",
            f"No CGM reading in {age_minutes:.0f} min — predictions are based on outdated data.",
            {"age_minutes": age_minutes, "last_seen": str(cur_ts)},
        ))

    # -- Forecast-based alerts. -----------------------------------------
    if prediction_30m < th["low_mg_dl"]:
        alerts.append(Alert(
            "predicted_low", "critical",
            f"Predicted {prediction_30m:.0f} mg/dL in 30 min (below {int(th['low_mg_dl'])}). "
            "Consider pre-treating.",
            {"prediction_30m": prediction_30m, "delta": delta_30m},
        ))
    elif prediction_30m > th["high_mg_dl"]:
        alerts.append(Alert(
            "predicted_high", "warn",
            f"Predicted {prediction_30m:.0f} mg/dL in 30 min (above {int(th['high_mg_dl'])}). "
            "Possible correction window.",
            {"prediction_30m": prediction_30m, "delta": delta_30m},
        ))

    if delta_30m <= -th["drop_mg_dl_30m"]:
        alerts.append(Alert(
            "predicted_drop", "warn",
            f"Predicted drop of {-delta_30m:.0f} mg/dL in the next 30 min. "
            "Watch for an oncoming low.",
            {"delta": delta_30m, "prediction_30m": prediction_30m},
        ))
    elif delta_30m >= th["rise_mg_dl_30m"]:
        alerts.append(Alert(
            "predicted_rise", "info",
            f"Predicted rise of {delta_30m:.0f} mg/dL in the next 30 min. "
            "Likely a carb-driven spike.",
            {"delta": delta_30m, "prediction_30m": prediction_30m},
        ))

    return alerts


def alerts_to_frame(alerts: Iterable[Alert]) -> pd.DataFrame:
    return pd.DataFrame([a.to_dict() for a in alerts]) if alerts else pd.DataFrame(
        columns=["type", "severity", "message", "detail"]
    )
