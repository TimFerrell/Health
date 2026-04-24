"""End-to-end smoke test: synthesize a Clarity-shaped CSV, run all
phases, and assert the unified parquet has the expected schema.

Not a unit test — just a quick canary so the pipeline can't silently
break when we touch shared code. Run with: python scripts/_smoke_test.py
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src import ingest_dexcom, ingest_tandem, merge_pipeline, validate_pipeline  # noqa: E402


def _fake_clarity_csv(path: Path, n_days: int = 3) -> None:
    end = datetime(2026, 4, 24, tzinfo=timezone.utc).replace(tzinfo=None)
    start = end - timedelta(days=n_days)
    ts = pd.date_range(start, end, freq="5min")
    rng = np.random.default_rng(0)
    glucose = 130 + 20 * np.sin(np.linspace(0, n_days * 2 * np.pi, len(ts))) + rng.normal(0, 8, len(ts))
    glucose = np.clip(glucose, 50, 280).round().astype(int)
    arrows = rng.choice(
        ["Flat", "FortyFiveUp", "FortyFiveDown", "SingleUp", "SingleDown"], size=len(ts)
    )
    df = pd.DataFrame(
        {
            "Index": np.arange(len(ts)),
            "Timestamp (YYYY-MM-DDThh:mm:ss)": ts.strftime("%Y-%m-%dT%H:%M:%S"),
            "Event Type": "EGV",
            "Event Subtype": "",
            "Patient Info": "",
            "Device Info": "",
            "Source Device ID": "G7",
            "Glucose Value (mg/dL)": glucose.astype(object),
            "Insulin Value (u)": "",
            "Carb Value (grams)": "",
            "Duration (hh:mm:ss)": "",
            "Glucose Rate of Change (mg/dL/min)": "",
            "Transmitter Time (Long Integer)": "",
            "Transmitter ID": "ABC123",
            "Trend Arrow": arrows,
        }
    )
    # Slip in a "Low" sentinel and a duplicate timestamp to test edge handling
    df.loc[5, "Glucose Value (mg/dL)"] = "Low"
    df.loc[6, "Glucose Value (mg/dL)"] = "High"
    dup = df.iloc[10:11].copy()
    df = pd.concat([df, dup], ignore_index=True)
    df.to_csv(path, index=False)


def main() -> int:
    out = REPO / "data" / "smoke"
    out.mkdir(parents=True, exist_ok=True)
    csv = out / "clarity_fake.csv"
    _fake_clarity_csv(csv)

    dx = ingest_dexcom.ingest(csv, out / "dexcom_clean.parquet")
    assert {"timestamp", "glucose_mg_dl", "trend_arrow_encoded"} <= set(dx.columns)
    assert dx["glucose_mg_dl"].between(39, 401).all()

    cfg = ingest_tandem.TandemConfig(
        email=None,
        password=None,
        start=dx["timestamp"].min().to_pydatetime(),
        end=dx["timestamp"].max().to_pydatetime(),
    )
    tx = ingest_tandem.ingest(cfg, out / "tandem_clean.parquet", allow_synthetic=True)
    assert set(tx["event_type"].unique()) <= set(ingest_tandem.EVENT_TYPES)

    merged = merge_pipeline.run(
        out / "dexcom_clean.parquet",
        out / "tandem_clean.parquet",
        out / "unified_timeline.parquet",
    )
    expected = {
        "timestamp",
        "glucose_mg_dl",
        "trend_arrow_encoded",
        "iob_units",
        "bolus_units",
        "basal_rate",
        "is_suspended",
        "control_iq_active",
    }
    assert set(merged.columns) == expected, f"schema mismatch: {set(merged.columns) ^ expected}"
    assert merged["is_suspended"].dtype == bool
    assert merged["control_iq_active"].dtype == bool
    assert (merged["iob_units"] >= 0).all()

    validate_pipeline.report(merged, out / "plots")
    print("\nSMOKE TEST: PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
