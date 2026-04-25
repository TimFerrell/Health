"""End-to-end smoke for the live pipeline:
   refresh -> anomaly -> counterfactual -> drift.
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

# Re-use the ML smoke to seed a trained model + parquets.
from scripts import _smoke_ml  # noqa: E402
from src import (  # noqa: E402
    anomaly,
    counterfactual,
    drift,
    features as feat_mod,
    predict as predict_mod,
    refresh as refresh_mod,
)


def main() -> int:
    out = REPO / "data" / "smoke_live"
    out.mkdir(parents=True, exist_ok=True)

    # 1. Bootstrap with the ML smoke (gives us a trained model + parquets).
    _smoke_ml.main()

    # Move artifacts to the live smoke dir and point refresh module at them.
    src_processed = REPO / "data" / "smoke_ml"
    src_models = src_processed / "models"

    refresh_mod.DATA_DIR = src_processed
    refresh_mod.MODEL_DIR = src_models
    refresh_mod.DEXCOM_PARQUET = src_processed / "dexcom_clean.parquet"
    refresh_mod.TANDEM_PARQUET = src_processed / "tandem_clean.parquet"
    refresh_mod.UNIFIED_PARQUET = src_processed / "unified_timeline.parquet"
    refresh_mod.FEATURES_PARQUET = src_processed / "features.parquet"
    refresh_mod.PREDICTIONS_LOG = src_processed / "predictions_log.parquet"
    refresh_mod.ALERTS_LOG = src_processed / "alerts_log.parquet"
    refresh_mod.METRICS_PATH = src_models / "metrics.json"

    # 2. Run a refresh cycle. We skip the pump pull because tconnectsync
    # would clobber our synthetic tandem parquet with a different range.
    result = refresh_mod.cycle(skip_pump=True)
    assert result["prediction"] is not None
    pred = result["prediction"]
    assert 40 <= pred["prediction_30m"] <= 401
    assert refresh_mod.PREDICTIONS_LOG.exists()

    # 3. Counterfactual on the latest row.
    feats = pd.read_parquet(refresh_mod.FEATURES_PARQUET)
    model, feature_cols = predict_mod.load_model(refresh_mod.MODEL_DIR)
    last_row = feats.dropna(subset=feature_cols).iloc[[-1]]
    cf = counterfactual.simulate(
        model, feature_cols, last_row,
        actions=counterfactual.standard_action_grid(),
    )
    assert {"baseline (no action)"} <= set(cf["scenario"])
    bolus_rows = cf[cf["scenario"].str.startswith("bolus")]
    # A bigger pretend bolus should reduce the prediction more than a
    # smaller one. (Sign check; magnitude depends on the model.)
    if len(bolus_rows) >= 2:
        assert bolus_rows.iloc[-1]["delta_vs_baseline"] <= bolus_rows.iloc[0]["delta_vs_baseline"], \
            "Bigger bolus should not predict higher than a smaller one"

    # 4. Anomaly detection: force a low scenario and verify alerts fire.
    fake_alerts = anomaly.detect(
        current_glucose=110.0,
        current_timestamp=pd.Timestamp.now(tz="UTC"),
        prediction_30m=58.0,
    )
    types = {a.type for a in fake_alerts}
    assert "predicted_low" in types
    assert "predicted_drop" in types

    # 5. Drift status — populate the log with a few synthetic predictions
    # and matching actuals so we can compute a real MAE.
    log_path = src_processed / "predictions_log.parquet"
    log_path.unlink(missing_ok=True)
    unified = pd.read_parquet(refresh_mod.UNIFIED_PARQUET)
    sample = unified.dropna(subset=["glucose_mg_dl"]).iloc[-200:-50]
    for _, row in sample.iterrows():
        ts = pd.Timestamp(row["timestamp"])
        target = ts + pd.Timedelta(minutes=30)
        actual_match = unified[unified["timestamp"] == target]
        if actual_match.empty:
            continue
        # "Predict" something close to actual + a noise term.
        actual = float(actual_match["glucose_mg_dl"].iloc[0])
        drift.append_log(log_path, {
            "predicted_at": ts,
            "target_timestamp": target,
            "current_glucose": float(row["glucose_mg_dl"]),
            "prediction_30m": actual + np.random.normal(0, 12),
        })
    status = drift.compute_status(log_path, refresh_mod.UNIFIED_PARQUET, refresh_mod.METRICS_PATH)
    assert status.n_scored > 0, "drift status should have at least one scored prediction"
    assert not np.isnan(status.mae_all)

    print("\nLIVE SMOKE TEST: PASSED")
    print(f"  refresh prediction: current={pred['current_glucose']:.0f} -> "
          f"{pred['prediction_30m']:.0f} (Δ {pred['predicted_change']:+.0f})")
    print(f"  alerts on synthetic low scenario: {sorted(types)}")
    print(f"  counterfactual scenarios: {len(cf)}")
    print(f"    sample row: {cf.iloc[1].to_dict()}")
    print(f"  drift: scored={status.n_scored}  mae_all={status.mae_all:.1f}  "
          f"trained_mae={status.trained_mae}  ratio={status.drift_ratio}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
