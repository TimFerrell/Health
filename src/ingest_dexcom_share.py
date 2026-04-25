"""Live Dexcom Share ingestion (~5 min latency).

The bulk Clarity CSV path (src/ingest_dexcom.py) is for backfill. This
module is the *live* path: it polls Dexcom's Share endpoint via
pydexcom for the most recent readings, normalizes them to the same
schema, and merges them into dexcom_clean.parquet, deduping on
timestamp.

Why Share instead of the official Dexcom API:
  * Share has ~5 minute latency. The official API is ~3 hours behind,
    which is useless for real-time prediction.
  * Share requires *follower* credentials (set up in the Dexcom mobile
    app under Share → invite a follower → use that follower's
    Dexcom account). Don't use the patient's primary login here.
  * Region matters: 'us', 'ous' (rest of world), or 'jp'.

If pydexcom isn't installed or credentials are missing, this module
generates synthetic readings so the rest of the live pipeline stays
exercisable.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# pydexcom returns trends as integers 0-7 in the order:
#   0=None, 1=DoubleUp, 2=SingleUp, 3=FortyFiveUp,
#   4=Flat, 5=FortyFiveDown, 6=SingleDown, 7=DoubleDown
# Convert to our project-standard -2..+2 scale.
PYDEXCOM_TREND_TO_ENCODED: dict[int, float] = {
    0: float("nan"),
    1: 2,   # double up
    2: 2,   # single up
    3: 1,   # forty-five up
    4: 0,   # flat
    5: -1,  # forty-five down
    6: -2,  # single down
    7: -2,  # double down
}


def fetch_via_pydexcom(minutes: int = 1440, max_count: int = 288) -> pd.DataFrame:
    """Pull the last `minutes` of glucose readings from Dexcom Share.

    Defaults: last 24 h (288 readings at 5-min cadence). Raises on any
    failure so the caller can decide whether to fall back to synthetic.
    """
    username = os.getenv("DEXCOM_USERNAME")
    password = os.getenv("DEXCOM_PASSWORD")
    region = os.getenv("DEXCOM_REGION", "us")
    if not username or not password:
        raise RuntimeError("DEXCOM_USERNAME / DEXCOM_PASSWORD not set in environment")

    from pydexcom import Dexcom  # type: ignore[import-not-found]

    dex = Dexcom(username=username, password=password, region=region)
    readings = dex.get_glucose_readings(minutes=minutes, max_count=max_count)

    rows: list[dict] = []
    for r in readings:
        rows.append(
            {
                "timestamp": pd.Timestamp(r.datetime).tz_convert("UTC")
                if pd.Timestamp(r.datetime).tz is not None
                else pd.Timestamp(r.datetime).tz_localize("UTC"),
                "glucose_mg_dl": float(r.value),
                "trend_arrow_encoded": PYDEXCOM_TREND_TO_ENCODED.get(int(r.trend or 0), float("nan")),
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("Dexcom Share returned no readings")
    return df


def synthesize_share(minutes: int = 1440, anchor: pd.Timestamp | None = None) -> pd.DataFrame:
    """Generate realistic-ish recent readings for offline development."""
    end = (anchor or pd.Timestamp.now(tz="UTC")).floor("5min")
    start = end - pd.Timedelta(minutes=minutes)
    ts = pd.date_range(start, end, freq="5min", tz="UTC")
    rng = np.random.default_rng(int(end.value) & 0xFFFFFFFF)

    hours = ts.hour + ts.minute / 60.0
    dawn = 22 * np.exp(-((hours - 5) ** 2) / 4.0)
    meal = (
        15 * np.exp(-((hours - 8) ** 2) / 1.5)
        + 18 * np.exp(-((hours - 12) ** 2) / 1.5)
        + 25 * np.exp(-((hours - 18) ** 2) / 1.5)
    )
    glucose = np.clip(115 + dawn + meal + rng.normal(0, 6, len(ts)), 55, 280).round()

    # Trend = sign of recent slope, clipped to -2..+2.
    slope = np.gradient(glucose) * 12  # convert per-bin to per-hour-ish
    trend = np.clip(np.round(slope / 30).astype(int), -2, 2)

    return pd.DataFrame({"timestamp": ts, "glucose_mg_dl": glucose, "trend_arrow_encoded": trend.astype(float)})


def merge_into_existing(new: pd.DataFrame, existing_path: Path) -> pd.DataFrame:
    """Append new readings to the canonical dexcom parquet, dedup on timestamp."""
    new = new.copy()
    new["timestamp"] = pd.to_datetime(new["timestamp"], utc=True)
    new["glucose_mg_dl"] = new["glucose_mg_dl"].astype("float32")
    new["trend_arrow_encoded"] = new["trend_arrow_encoded"].astype("float32")

    if existing_path.exists():
        existing = pd.read_parquet(existing_path)
        combined = pd.concat([existing, new], ignore_index=True)
    else:
        combined = new

    combined = combined.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last").reset_index(drop=True)
    existing_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(existing_path, index=False)
    return combined


def ingest(
    existing_path: Path,
    minutes: int = 1440,
    allow_synthetic: bool = True,
) -> tuple[pd.DataFrame, str]:
    try:
        new = fetch_via_pydexcom(minutes=minutes)
        source = "pydexcom"
    except Exception as exc:  # noqa: BLE001
        if not allow_synthetic:
            raise
        logger.warning("Dexcom Share fetch failed (%s); using synthetic readings", exc)
        new = synthesize_share(minutes=minutes)
        source = "synthetic"

    combined = merge_into_existing(new, existing_path)
    logger.info(
        "Merged %d new readings (source=%s); dexcom_clean now has %d rows up to %s",
        len(new), source, len(combined), combined["timestamp"].max(),
    )
    return combined, source


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Pull recent CGM readings from Dexcom Share.")
    p.add_argument("--minutes", type=int, default=1440, help="How many minutes back to pull (default 24h).")
    p.add_argument("-o", "--output", type=Path, default=Path("data/processed/dexcom_clean.parquet"))
    p.add_argument("--no-synthetic", action="store_true")
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass
    ingest(args.output, minutes=args.minutes, allow_synthetic=not args.no_synthetic)
    return 0


if __name__ == "__main__":
    sys.exit(main())
