"""Manual carb-treatment log.

Why this exists: tconnectsync gives us carbs *that were entered into
the pump's bolus calculator*. For a child, that misses a huge chunk:

  * Low treatments (juice, glucose tabs) — pure carbs, no bolus.
  * Unbolused snacks (pre-bedtime fruit, cheese sticks).
  * Mid-correction snacks where no bolus was given.

This module is the simplest possible append-only log for those events.
The schema deliberately mirrors a Nightscout `treatments` document so
a future Phase can swap in a Nightscout client without changing the
downstream pipeline.

Schema (treatments.parquet):
    timestamp        UTC, ISO 8601
    kind             "meal" | "low_treatment" | "snack" | "correction" | "other"
    carbs_g          grams (float32, never NaN — that's the whole point)
    notes            free text (e.g. "juice box at recess")
    source           "manual" | "tandem_bolus" | "nightscout"
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd

logger = logging.getLogger(__name__)

SCHEMA = ["timestamp", "kind", "carbs_g", "notes", "source"]
KINDS = ("meal", "low_treatment", "snack", "correction", "other")


def empty_frame() -> pd.DataFrame:
    return pd.DataFrame({
        "timestamp": pd.Series(dtype="datetime64[ns, UTC]"),
        "kind": pd.Series(dtype="object"),
        "carbs_g": pd.Series(dtype="float32"),
        "notes": pd.Series(dtype="object"),
        "source": pd.Series(dtype="object"),
    })


def load(path: Path) -> pd.DataFrame:
    if not path.exists():
        return empty_frame()
    df = pd.read_parquet(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["carbs_g"] = pd.to_numeric(df["carbs_g"], errors="coerce").astype("float32")
    return df.sort_values("timestamp").reset_index(drop=True)


def save(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df[SCHEMA]
    df.to_parquet(path, index=False)


def append(
    path: Path,
    *,
    timestamp: datetime | pd.Timestamp | None = None,
    kind: str = "meal",
    carbs_g: float = 0.0,
    notes: str = "",
    source: str = "manual",
) -> pd.DataFrame:
    """Append one treatment row. Returns the full updated dataframe."""
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")
    if carbs_g < 0:
        raise ValueError("carbs_g must be non-negative")

    ts = pd.Timestamp(timestamp) if timestamp is not None else pd.Timestamp.now(tz="UTC")
    if ts.tz is None:
        ts = ts.tz_localize("UTC")

    existing = load(path)
    new_row = pd.DataFrame([{
        "timestamp": ts,
        "kind": kind,
        "carbs_g": float(carbs_g),
        "notes": notes,
        "source": source,
    }])
    combined = pd.concat([existing, new_row], ignore_index=True)
    combined = combined.sort_values("timestamp").reset_index(drop=True)
    save(combined, path)
    logger.info("Logged treatment: %s %.0f g (%s) at %s", kind, carbs_g, source, ts)
    return combined


def derive_from_tandem(tandem: pd.DataFrame) -> pd.DataFrame:
    """Pull bolus-attached carbs out of tandem_clean.parquet as treatments.

    These are the carbs the user entered into the pump's bolus
    calculator. Any bolus row without carbs_g is dropped (it was a
    quick bolus / manual override).
    """
    if tandem.empty or "carbs_g" not in tandem.columns:
        return empty_frame()
    bolus = tandem[(tandem["event_type"] == "bolus") & tandem["carbs_g"].notna() & (tandem["carbs_g"] > 0)]
    if bolus.empty:
        return empty_frame()
    return pd.DataFrame({
        "timestamp": pd.to_datetime(bolus["timestamp"], utc=True),
        "kind": "meal",
        "carbs_g": bolus["carbs_g"].astype("float32"),
        "notes": "",
        "source": "tandem_bolus",
    }).reset_index(drop=True)


def union(*frames: pd.DataFrame) -> pd.DataFrame:
    """Concatenate treatment frames (manual + tandem-derived + future
    Nightscout) and dedup. Dedup key: (timestamp, source, kind, carbs_g)
    so the same meal logged twice (once via pump, once manually) gets
    merged sensibly without losing either signal."""
    non_empty = [f for f in frames if f is not None and not f.empty]
    if not non_empty:
        return empty_frame()
    combined = pd.concat(non_empty, ignore_index=True)
    combined["timestamp"] = pd.to_datetime(combined["timestamp"], utc=True)
    combined = combined.drop_duplicates(
        subset=["timestamp", "source", "kind", "carbs_g"],
    ).sort_values("timestamp").reset_index(drop=True)
    return combined


# Quick-action presets for the UI's low-treatment buttons. Sized for
# pediatric use; adults typically scale up by ~1.5x.
LOW_TREATMENT_PRESETS = [
    {"label": "4 g (gel)",       "carbs_g": 4,  "notes": "glucose gel"},
    {"label": "8 g (juice box)", "carbs_g": 8,  "notes": "juice box"},
    {"label": "15 g (standard)", "carbs_g": 15, "notes": "standard low treatment"},
]
