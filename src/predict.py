"""Apply a trained model to produce glucose predictions.

For non-ML readers: training (src/train.py) is "learning the patterns".
Prediction is "using what was learned to make a guess on new data".
This script is what you'd call in real-time inference once we ship.

Two modes:
  --row LATEST   : predict glucose 30 min ahead of the latest available
                   reading. Useful for ad-hoc checks.
  --all          : predict for every row in the input table that has a
                   complete feature set. Useful for backtesting and the
                   "predicted vs actual" UI chart.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

logger = logging.getLogger(__name__)


def load_model(model_dir: Path) -> tuple[xgb.XGBRegressor, list[str]]:
    """Reload the saved model + the exact feature column order.

    The column order matters: XGBoost identifies features by position
    inside the booster. If we passed columns in a different order at
    inference time, every prediction would silently be garbage.
    """
    model = xgb.XGBRegressor()
    model.load_model(model_dir / "xgb_glucose_30m.json")
    feature_cols: list[str] = json.loads((model_dir / "feature_columns.json").read_text())
    return model, feature_cols


def predict_dataframe(model: xgb.XGBRegressor, feature_cols: list[str], features_df: pd.DataFrame) -> pd.DataFrame:
    """Predict for every row that has a non-NaN feature vector.

    Returns a dataframe with timestamp, current glucose, prediction,
    and (if the truth is known) the actual future glucose for backtest
    comparison.
    """
    df = features_df.copy()
    # Drop rows with any NaN in the feature columns. XGBoost actually
    # tolerates NaNs natively, but for backtest plots we want
    # apples-to-apples coverage with the training data, which itself
    # had targets attached.
    feature_view = df[feature_cols]
    df["prediction_30m"] = model.predict(feature_view)
    out = df[["timestamp", "glucose_mg_dl", "prediction_30m"]].copy()
    if "target_30m" in df.columns:
        out["actual_30m"] = df["target_30m"]
        out["abs_error"] = (out["prediction_30m"] - out["actual_30m"]).abs()
    return out


def predict_latest(model: xgb.XGBRegressor, feature_cols: list[str], features_df: pd.DataFrame) -> dict:
    """Predict 30 min ahead from the most recent fully-populated row."""
    sub = features_df.dropna(subset=feature_cols)
    if sub.empty:
        raise ValueError("No row with a complete feature vector — pipeline produced too little context.")
    last = sub.iloc[[-1]]
    pred = float(model.predict(last[feature_cols])[0])
    return {
        "as_of": str(last["timestamp"].iloc[0]),
        "current_glucose": float(last["glucose_mg_dl"].iloc[0]),
        "prediction_30m": pred,
        "predicted_change": pred - float(last["glucose_mg_dl"].iloc[0]),
    }


def run(features_path: Path, model_dir: Path, output_path: Path | None, latest_only: bool) -> object:
    df = pd.read_parquet(features_path)
    model, feature_cols = load_model(model_dir)

    if latest_only:
        result = predict_latest(model, feature_cols, df)
        logger.info(
            "As of %s: current=%.0f mg/dL, predicted in 30 min=%.0f (Δ %+0.0f)",
            result["as_of"], result["current_glucose"], result["prediction_30m"], result["predicted_change"],
        )
        return result

    out = predict_dataframe(model, feature_cols, df)
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        out.to_parquet(output_path, index=False)
        logger.info("Wrote %d predictions to %s", len(out), output_path)
    if "abs_error" in out.columns:
        # Same metric as train.py reports — should match closely if
        # there's no leakage. (It will be slightly different because
        # this includes train+val rows the model has seen.)
        logger.info("Backtest MAE across all rows: %.2f mg/dL", out["abs_error"].mean())
    return out


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Predict glucose with the trained XGBoost model.")
    p.add_argument("--features", type=Path, default=Path("data/processed/features.parquet"))
    p.add_argument("--model-dir", type=Path, default=Path("data/models"))
    p.add_argument("-o", "--output", type=Path, default=Path("data/processed/predictions.parquet"))
    p.add_argument("--latest", action="store_true", help="Print only the most recent prediction.")
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    if not args.features.exists():
        logger.error("Missing features parquet: %s", args.features)
        return 2
    output = None if args.latest else args.output
    run(args.features, args.model_dir, output, latest_only=args.latest)
    return 0


if __name__ == "__main__":
    sys.exit(main())
