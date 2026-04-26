"""Pull therapy timeline from a Tandem Mobi pump via tconnectsync.

The Tandem t:connect API is undocumented; tconnectsync is a
reverse-engineered client that frequently breaks when Tandem rotates
auth or their CIQ endpoints. To keep the rest of the pipeline testable
when that happens, this module falls back to a synthetic generator that
mimics realistic basal/bolus/Control-IQ patterns for a T1D child.

Why this matters for prediction:
- Boluses are the dominant short-horizon glucose driver (insulin on board
  decays roughly exponentially with a ~4 hour half-life for rapid-acting
  analogs like Humalog/Novolog/Fiasp used in Mobi).
- Control-IQ "suspend" events (auto-suspend by the closed loop on
  predicted lows) are a strong predictor of an upcoming low and a
  subsequent rebound — even if the low never materializes in CGM data.
- Basal rate changes from Control-IQ "increase" actions tell us the
  algorithm is reacting to a predicted high; useful as a secondary
  signal that the pump "sees" what the model should also see.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

EVENT_TYPES = ("basal", "bolus", "suspend", "control_iq_action")


@dataclass
class TandemConfig:
    email: str | None
    password: str | None
    start: datetime
    end: datetime

    @classmethod
    def from_env(cls, start: datetime | None = None, end: datetime | None = None) -> "TandemConfig":
        end = end or datetime.now(tz=timezone.utc)
        start = start or (end - timedelta(days=30))
        return cls(
            email=os.getenv("TCONNECT_EMAIL"),
            password=os.getenv("TCONNECT_PASSWORD"),
            start=start,
            end=end,
        )


def _to_utc(ts: object) -> pd.Timestamp:
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        return t.tz_localize("UTC")
    return t.tz_convert("UTC")


def fetch_via_tconnectsync(cfg: TandemConfig) -> pd.DataFrame:
    """Best-effort call into tconnectsync. Raises on any failure so the
    caller can decide whether to fall back to synthetic data."""
    if not cfg.email or not cfg.password:
        raise RuntimeError("TCONNECT_EMAIL / TCONNECT_PASSWORD not set in environment")

    # Imports are local so the synthetic path works even if tconnectsync
    # isn't installed in the current environment.
    from tconnectsync.api import TConnectApi  # type: ignore[import-not-found]

    api = TConnectApi(cfg.email, cfg.password)

    # tconnectsync exposes several event streams. We pull the ones that
    # map to our four canonical event types and concatenate them.
    rows: list[dict] = []

    # Bolus events (carb + correction). t:connect's BolusRequestDTO
    # carries the carbs the user entered into the calculator and the
    # split between food / correction insulin. Field names vary slightly
    # by tconnectsync version, so we probe a handful of candidates.
    for b in api.controliq.dailybolusdata(cfg.start, cfg.end):
        carbs_g = b.get("carbsRequest") or b.get("carbsAmount") or b.get("carbs") or b.get("carbAmount")
        bg = b.get("bgValue") or b.get("bg") or b.get("targetBG")
        food_u = b.get("foodInsulin") or b.get("foodInsulinDelivered")
        corr_u = b.get("correctionInsulin") or b.get("correctionInsulinDelivered")
        rows.append(
            {
                "timestamp": _to_utc(b.get("eventDateTime") or b.get("requestDateTime")),
                "event_type": "bolus",
                "value": float(b.get("insulinDelivered") or b.get("requestedInsulin") or 0.0),
                "duration_minutes": float(b.get("duration") or 0.0) / 60.0 if b.get("duration") else 0.0,
                "iob_units": float(b.get("iob")) if b.get("iob") is not None else float("nan"),
                # New optional carb-context fields. NaN when the bolus
                # was a manual override or the calculator wasn't used.
                "carbs_g": float(carbs_g) if carbs_g is not None else float("nan"),
                "bg_at_bolus": float(bg) if bg is not None else float("nan"),
                "food_insulin": float(food_u) if food_u is not None else float("nan"),
                "correction_insulin": float(corr_u) if corr_u is not None else float("nan"),
            }
        )

    # Basal rate samples (one row per CIQ sampling interval)
    for r in api.controliq.dailybasaldata(cfg.start, cfg.end):
        rows.append(
            {
                "timestamp": _to_utc(r.get("eventDateTime")),
                "event_type": "basal",
                "value": float(r.get("basalRate") or 0.0),
                "duration_minutes": float(r.get("duration") or 5.0),
                "iob_units": float("nan"),
            }
        )

    # Suspend / resume + CIQ algorithm actions
    for e in api.controliq.dailyeventdata(cfg.start, cfg.end):
        et = (e.get("eventType") or "").lower()
        kind = "suspend" if "suspend" in et else "control_iq_action"
        rows.append(
            {
                "timestamp": _to_utc(e.get("eventDateTime")),
                "event_type": kind,
                "value": float(e.get("value") or 0.0),
                "duration_minutes": float(e.get("duration") or 0.0),
                "iob_units": float("nan"),
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("tconnectsync returned no events for the requested window")
    return df


def synthesize_tandem(cfg: TandemConfig, seed: int = 42) -> pd.DataFrame:
    """Generate plausible pump data so downstream phases stay testable.

    Patterns modeled:
      * Background basal cycling (~0.4-0.7 U/hr, higher overnight for the
        pre-dawn rise typical in growing kids).
      * Three meal boluses per day (breakfast/lunch/dinner) with
        weekday-correlated timing jitter.
      * Occasional Control-IQ auto-suspends overnight.
      * Periodic CIQ "increase basal" actions in the afternoon.
    """
    rng = np.random.default_rng(seed)
    rows: list[dict] = []

    day = cfg.start.replace(hour=0, minute=0, second=0, microsecond=0)
    end = cfg.end
    while day <= end:
        # Hourly basal samples for the day
        for hour in range(24):
            t = day + timedelta(hours=hour)
            if t < cfg.start or t > cfg.end:
                continue
            base = 0.45 if 6 <= hour < 22 else 0.65  # higher overnight
            jitter = rng.normal(0, 0.05)
            rows.append(
                {
                    "timestamp": _to_utc(t),
                    "event_type": "basal",
                    "value": max(0.0, round(base + jitter, 3)),
                    "duration_minutes": 60.0,
                    "iob_units": float("nan"),
                }
            )

        # Three meal boluses with realistic dosing for a child (~0.5-3 U).
        # We back-compute carbs from the dose using a child-typical ICR
        # (insulin-to-carb ratio) of 12 g per U, with some jitter.
        for meal_hour, mean_dose in ((7.5, 2.5), (12.0, 2.0), (18.0, 3.0)):
            t = day + timedelta(hours=meal_hour) + timedelta(minutes=int(rng.uniform(-20, 20)))
            if t < cfg.start or t > cfg.end:
                continue
            dose = max(0.1, round(rng.normal(mean_dose, 0.5), 2))
            icr = max(8.0, rng.normal(12.0, 1.5))
            carbs = round(dose * icr, 1)
            rows.append(
                {
                    "timestamp": _to_utc(t),
                    "event_type": "bolus",
                    "value": dose,
                    "duration_minutes": 0.0,
                    # Synthetic IOB starts at the dose itself; merge layer
                    # applies decay if the raw value isn't trusted.
                    "iob_units": dose,
                    "carbs_g": carbs,
                    "bg_at_bolus": float(round(rng.normal(140, 25))),
                    "food_insulin": dose * 0.85,
                    "correction_insulin": dose * 0.15,
                }
            )

        # ~30% chance of an overnight CIQ suspend
        if rng.random() < 0.3:
            t = day + timedelta(hours=int(rng.uniform(2, 5)), minutes=int(rng.uniform(0, 59)))
            if cfg.start <= t <= cfg.end:
                rows.append(
                    {
                        "timestamp": _to_utc(t),
                        "event_type": "suspend",
                        "value": 0.0,
                        "duration_minutes": float(rng.uniform(10, 45)),
                        "iob_units": float("nan"),
                    }
                )

        # Afternoon CIQ "increase basal" reaction
        if rng.random() < 0.5:
            t = day + timedelta(hours=int(rng.uniform(14, 17)), minutes=int(rng.uniform(0, 59)))
            if cfg.start <= t <= cfg.end:
                rows.append(
                    {
                        "timestamp": _to_utc(t),
                        "event_type": "control_iq_action",
                        "value": 1.0,  # boolean: action fired
                        "duration_minutes": 30.0,
                        "iob_units": float("nan"),
                    }
                )

        day += timedelta(days=1)

    df = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)
    return df


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce columns into the canonical schema and types.

    Carb-context columns are optional and only meaningful on bolus
    rows; we fill NaN for everything else. Backward-compatible with
    parquets written before these columns existed.
    """
    required = ["timestamp", "event_type", "value", "duration_minutes", "iob_units"]
    optional_floats = ["carbs_g", "bg_at_bolus", "food_insulin", "correction_insulin"]
    for col in required:
        if col not in df.columns:
            df[col] = float("nan") if col != "event_type" else ""
    for col in optional_floats:
        if col not in df.columns:
            df[col] = float("nan")
    df = df[required + optional_floats].copy()
    df["event_type"] = df["event_type"].astype("category")
    for col in ["value", "duration_minutes", "iob_units", *optional_floats]:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("float32")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def ingest(cfg: TandemConfig, output_path: Path, allow_synthetic: bool = True) -> pd.DataFrame:
    try:
        raw = fetch_via_tconnectsync(cfg)
        source = "tconnectsync"
    except Exception as exc:  # noqa: BLE001 — we genuinely want to catch anything
        if not allow_synthetic:
            raise
        logger.warning("tconnectsync ingestion failed (%s); falling back to synthetic data", exc)
        raw = synthesize_tandem(cfg)
        source = "synthetic"

    cleaned = normalize(raw)
    if cleaned["iob_units"].isna().all():
        logger.warning("No IOB values present in source=%s — merge layer will model decay synthetically.", source)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_parquet(output_path, index=False)
    logger.info(
        "Wrote %d events from %s to %s (range %s -> %s)",
        len(cleaned),
        source,
        output_path,
        cleaned["timestamp"].min(),
        cleaned["timestamp"].max(),
    )
    return cleaned


def _parse_date(s: str) -> datetime:
    dt = pd.Timestamp(s).to_pydatetime()
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Ingest Tandem pump events via tconnectsync.")
    p.add_argument("--start", type=_parse_date, default=None, help="Start date (ISO 8601, default 30d ago)")
    p.add_argument("--end", type=_parse_date, default=None, help="End date (ISO 8601, default now)")
    p.add_argument("-o", "--output", type=Path, default=Path("data/processed/tandem_clean.parquet"))
    p.add_argument("--no-synthetic", action="store_true", help="Fail instead of falling back to synthetic data")
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def main(argv: Iterable[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(list(argv) if argv is not None else None)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        logger.debug("python-dotenv not installed; relying on shell env")

    # Honor env defaults if CLI didn't override
    start = args.start or (_parse_date(os.environ["TANDEM_START_DATE"]) if os.getenv("TANDEM_START_DATE") else None)
    end = args.end or (_parse_date(os.environ["TANDEM_END_DATE"]) if os.getenv("TANDEM_END_DATE") else None)
    cfg = TandemConfig.from_env(start=start, end=end)

    ingest(cfg, args.output, allow_synthetic=not args.no_synthetic)
    return 0


if __name__ == "__main__":
    sys.exit(main())
