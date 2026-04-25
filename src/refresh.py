"""Live refresh orchestrator.

One refresh cycle:
  1. Pull recent CGM readings from Dexcom Share (or synthetic).
  2. Pull recent pump events from t:connect (or synthetic).
  3. Re-merge → re-feature the *tail* of the timeline.
  4. Predict glucose 30 min ahead from the latest fully-populated row.
  5. Append the prediction to the log (drift.py uses this).
  6. Run anomaly detection on the latest prediction.
  7. Print the result (and return it for callers).

Run modes:
  python -m src.refresh                # one cycle, exit
  python -m src.refresh --loop --interval 300   # every 5 minutes, forever

The loop mode is what the docker-compose scheduler sidecar uses.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from src import (
    anomaly,
    drift,
    features as feat_mod,
    ingest_dexcom_share,
    ingest_tandem,
    merge_pipeline,
    predict as predict_mod,
)

logger = logging.getLogger(__name__)

DATA_DIR = Path("data/processed")
MODEL_DIR = Path("data/models")
DEXCOM_PARQUET = DATA_DIR / "dexcom_clean.parquet"
TANDEM_PARQUET = DATA_DIR / "tandem_clean.parquet"
UNIFIED_PARQUET = DATA_DIR / "unified_timeline.parquet"
FEATURES_PARQUET = DATA_DIR / "features.parquet"
PREDICTIONS_LOG = DATA_DIR / "predictions_log.parquet"
ALERTS_LOG = DATA_DIR / "alerts_log.parquet"
METRICS_PATH = MODEL_DIR / "metrics.json"


def _append_alerts(alerts: list[anomaly.Alert], when: pd.Timestamp) -> None:
    if not alerts:
        return
    rows = [{"timestamp": when, **a.to_dict()} for a in alerts]
    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["detail"] = df["detail"].apply(json.dumps)
    if ALERTS_LOG.exists():
        existing = pd.read_parquet(ALERTS_LOG)
        df = pd.concat([existing, df], ignore_index=True)
    df = df.tail(1000)  # cap so the file doesn't grow forever
    ALERTS_LOG.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(ALERTS_LOG, index=False)


def cycle(
    *,
    dexcom_minutes: int = 1440,
    tandem_days: int = 7,
    allow_synthetic: bool = True,
    skip_pump: bool = False,
) -> dict:
    """Run one refresh cycle. Returns a dict with the latest prediction
    and any alerts raised."""

    # 1. CGM
    ingest_dexcom_share.ingest(DEXCOM_PARQUET, minutes=dexcom_minutes, allow_synthetic=allow_synthetic)

    # 2. Pump (skip option for test scenarios where we already loaded
    # synthetic pump data manually).
    if not skip_pump:
        end = datetime.now(tz=timezone.utc)
        start = end - timedelta(days=tandem_days)
        cfg = ingest_tandem.TandemConfig.from_env(start=start, end=end)
        ingest_tandem.ingest(cfg, TANDEM_PARQUET, allow_synthetic=allow_synthetic)

    # 3. Merge + features (full re-build; 60 days fits in <1s on the
    # merged timeline. Tail-only re-features would be a future opt.)
    merge_pipeline.run(DEXCOM_PARQUET, TANDEM_PARQUET, UNIFIED_PARQUET)
    feat_mod.run(UNIFIED_PARQUET, FEATURES_PARQUET)

    # 4. Predict from the latest fully-populated row.
    if not (MODEL_DIR / "xgb_glucose_30m.json").exists():
        logger.warning("No trained model present; skipping prediction stage.")
        return {"prediction": None, "alerts": []}

    feats = pd.read_parquet(FEATURES_PARQUET)
    model, feature_cols = predict_mod.load_model(MODEL_DIR)
    latest = predict_mod.predict_latest(model, feature_cols, feats)

    predicted_at = pd.Timestamp(latest["as_of"])
    target_ts = predicted_at + pd.Timedelta(minutes=feat_mod.HORIZON_MIN)

    # 5. Append to the predictions log (drift uses this).
    drift.append_log(PREDICTIONS_LOG, {
        "predicted_at": predicted_at,
        "target_timestamp": target_ts,
        "current_glucose": latest["current_glucose"],
        "prediction_30m": latest["prediction_30m"],
    })

    # 6. Anomalies.
    alerts = anomaly.detect(
        current_glucose=latest["current_glucose"],
        current_timestamp=predicted_at,
        prediction_30m=latest["prediction_30m"],
    )
    _append_alerts(alerts, pd.Timestamp.now(tz="UTC"))

    return {"prediction": latest, "alerts": [a.to_dict() for a in alerts]}


def _summarize(result: dict) -> str:
    pred = result.get("prediction")
    if not pred:
        return "(no prediction — model not trained yet)"
    line = (
        f"as_of={pred['as_of']}  current={pred['current_glucose']:.0f}  "
        f"predict_30m={pred['prediction_30m']:.0f}  "
        f"Δ={pred['predicted_change']:+.0f}"
    )
    if result["alerts"]:
        line += "  alerts: " + ", ".join(f"{a['severity']}:{a['type']}" for a in result["alerts"])
    return line


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run one (or many) live refresh cycles.")
    p.add_argument("--loop", action="store_true", help="Run forever, sleeping between cycles.")
    p.add_argument("--interval", type=int, default=300, help="Seconds between cycles in --loop mode.")
    p.add_argument("--dexcom-minutes", type=int, default=1440)
    p.add_argument("--tandem-days", type=int, default=7)
    p.add_argument("--no-synthetic", action="store_true")
    p.add_argument("--skip-pump", action="store_true",
                   help="Skip the t:connect pull (useful when you already have synthetic pump data).")
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

    def run_one():
        result = cycle(
            dexcom_minutes=args.dexcom_minutes,
            tandem_days=args.tandem_days,
            allow_synthetic=not args.no_synthetic,
            skip_pump=args.skip_pump,
        )
        logger.info(_summarize(result))
        return result

    if not args.loop:
        run_one()
        return 0

    logger.info("Refresh loop starting (interval=%ds).", args.interval)
    while True:
        try:
            run_one()
        except KeyboardInterrupt:
            logger.info("Interrupted; exiting.")
            return 0
        except Exception:  # noqa: BLE001
            logger.exception("Refresh cycle failed; will retry next interval.")
        time.sleep(args.interval)


if __name__ == "__main__":
    sys.exit(main())
