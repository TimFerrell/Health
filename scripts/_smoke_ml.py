"""End-to-end smoke test for the ML pipeline.

Synthesizes a longer Dexcom + Tandem run (so train/val/test all have
data), runs features -> train -> predict -> explain, and asserts
sane outputs.
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src import (  # noqa: E402
    explain,
    features as feat_mod,
    ingest_dexcom,
    ingest_tandem,
    merge_pipeline,
    predict as predict_mod,
    train as train_mod,
)


def _fake_clarity_csv(path: Path, n_days: int = 60) -> None:
    end = datetime(2026, 4, 24, tzinfo=timezone.utc).replace(tzinfo=None)
    start = end - timedelta(days=n_days)
    ts = pd.date_range(start, end, freq="5min")
    rng = np.random.default_rng(0)
    # Layered signal: dawn rise + meal spikes + noise
    hours = (ts.hour + ts.minute / 60.0)
    dawn = 25 * np.exp(-((hours - 5) ** 2) / 4.0)
    meal = (
        18 * np.exp(-((hours - 8) ** 2) / 1.5)
        + 22 * np.exp(-((hours - 12) ** 2) / 1.5)
        + 28 * np.exp(-((hours - 18) ** 2) / 1.5)
    )
    noise = rng.normal(0, 6, len(ts))
    glucose = 110 + dawn + meal + noise
    glucose = np.clip(glucose, 55, 280).round().astype(int)

    df = pd.DataFrame(
        {
            "Index": np.arange(len(ts)),
            "Timestamp (YYYY-MM-DDThh:mm:ss)": ts.strftime("%Y-%m-%dT%H:%M:%S"),
            "Event Type": "EGV",
            "Patient Info": "",
            "Source Device ID": "G7",
            "Glucose Value (mg/dL)": glucose.astype(object),
            "Trend Arrow": rng.choice(["Flat", "FortyFiveUp", "FortyFiveDown", "SingleUp", "SingleDown"], size=len(ts)),
        }
    )
    df.to_csv(path, index=False)


def main() -> int:
    out = REPO / "data" / "smoke_ml"
    out.mkdir(parents=True, exist_ok=True)
    csv = out / "clarity_fake.csv"
    _fake_clarity_csv(csv)

    dexcom_pq = out / "dexcom_clean.parquet"
    tandem_pq = out / "tandem_clean.parquet"
    unified_pq = out / "unified_timeline.parquet"
    features_pq = out / "features.parquet"
    model_dir = out / "models"
    pred_pq = out / "predictions.parquet"

    ingest_dexcom.ingest(csv, dexcom_pq)

    cfg = ingest_tandem.TandemConfig(
        email=None, password=None,
        start=datetime(2026, 2, 24, tzinfo=timezone.utc),
        end=datetime(2026, 4, 24, tzinfo=timezone.utc),
    )
    ingest_tandem.ingest(cfg, tandem_pq, allow_synthetic=True)

    merge_pipeline.run(dexcom_pq, tandem_pq, unified_pq)
    feat_mod.run(unified_pq, features_pq)

    # Carbs should have flowed end-to-end if Tandem provided them.
    feats_check = pd.read_parquet(features_pq)
    carb_cols = [c for c in feats_check.columns if c.startswith("carbs_sum_") or c == "minutes_since_carbs"]
    assert carb_cols, f"expected carb features in feature table, got {list(feats_check.columns)[:6]}..."
    assert (feats_check["carbs_sum_60m"] > 0).any(), "carbs_sum_60m never non-zero — pump-derived carbs missing"

    result = train_mod.run(features_pq, model_dir)
    m = result["metrics"]
    assert m["mae"] < 40, f"MAE suspiciously high: {m['mae']}"
    assert m["mae"] > 1, f"MAE suspiciously low (leak?): {m['mae']}"
    assert m["test_rows"] > 100

    preds = predict_mod.run(features_pq, model_dir, pred_pq, latest_only=False)
    assert "prediction_30m" in preds.columns
    latest = predict_mod.run(features_pq, model_dir, None, latest_only=True)
    assert 40 <= latest["prediction_30m"] <= 401

    expl = explain.run(features_pq, model_dir, model_dir, sample_size=500)
    importance = expl["importance"]
    assert len(importance) > 0
    top_feature = importance.iloc[0]["feature"]
    # Sanity: a recent-glucose lag should dominate. If it doesn't,
    # something is broken in feature construction.
    assert "glucose" in top_feature, f"Top feature isn't glucose-derived: {top_feature}"

    print(f"\nML SMOKE TEST: PASSED")
    print(f"  test MAE      : {m['mae']:.2f} mg/dL")
    print(f"  test RMSE     : {m['rmse']:.2f}")
    print(f"  within ±20    : {m['pct_within_20']:.1f}%")
    print(f"  hypo recall   : {m['hypo_recall']}")
    print(f"  top feature   : {top_feature}  (mean|SHAP|={importance.iloc[0]['mean_abs_shap']:.2f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
