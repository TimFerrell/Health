"""SHAP-based explanations for the trained model.

For non-ML readers: see the SHAP section of docs/ML_PRIMER.md. The TL;DR:

  We computed a single number — "predicted glucose in 30 min". SHAP
  decomposes that number into per-feature contributions:

      "We predicted 175 mg/dL because:
          baseline                               +110
          glucose right now (165 mg/dL)           +35
          rising trend (+1)                       +25
          IOB is 1.4 U                             -8
          hour of day = 8am                        +6
          ... etc ..."

  The contributions sum to the prediction (after a baseline offset).

This file produces:
  * Global feature importance (which features the model leans on overall)
  * Per-row local explanations (why this *specific* prediction)
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src import predict as predict_mod

logger = logging.getLogger(__name__)


def _shap_values(model, X: pd.DataFrame) -> np.ndarray:
    """Compute SHAP values via XGBoost's built-in TreeSHAP.

    XGBoost ships TreeSHAP — exact, fast, no external `shap` package
    required. Each row in the returned array is one SHAP value per
    feature plus a final "bias" term.

    Returns a 2-D array of shape (n_rows, n_features + 1). Some
    XGBoost versions return 1-D for n_rows == 1; we always reshape to
    2-D so callers don't need to special-case it.
    """
    booster = model.get_booster()
    import xgboost as xgb  # local import keeps top of file lean

    dm = xgb.DMatrix(X)
    contribs = np.asarray(booster.predict(dm, pred_contribs=True))
    if contribs.ndim == 1:
        # xgboost 2.x with single-row input returns (n_features+1,).
        contribs = contribs.reshape(1, -1)
    elif contribs.ndim == 3:
        # Multi-class classifier path: (n_rows, n_classes, n_features+1).
        # Not our case (regression), but defensive.
        contribs = contribs.reshape(contribs.shape[0], -1)
    return contribs


def global_importance(model, X: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """Rank features by mean(|SHAP value|) across the sample.

    "How much, on average, did this feature move the prediction in
    either direction?" — a more honest measure than gain-based
    importance, which can over-weight features split early.
    """
    contribs = _shap_values(model, X)
    # Last column is the bias term; drop it.
    feat_contribs = contribs[:, :-1]
    mean_abs = np.abs(feat_contribs).mean(axis=0)
    out = pd.DataFrame({"feature": feature_cols, "mean_abs_shap": mean_abs})
    out = out.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    return out


def local_explanation(model, X_row: pd.DataFrame, feature_cols: list[str], top_k: int = 8) -> dict:
    """Why did the model predict X for this single row.

    Returns the baseline, the prediction, and the top-k features by
    absolute contribution (positive and negative both included).
    """
    contribs = _shap_values(model, X_row)  # always (n_rows, n_features + 1)
    feat_contribs = contribs[0, :-1]
    bias = float(contribs[0, -1])
    pairs = sorted(zip(feature_cols, feat_contribs), key=lambda kv: abs(kv[1]), reverse=True)[:top_k]
    return {
        "baseline": bias,
        "prediction": bias + float(feat_contribs.sum()),
        "contributions": [{"feature": f, "value": float(X_row.iloc[0][f]), "shap": float(s)} for f, s in pairs],
    }


def run(features_path: Path, model_dir: Path, output_dir: Path, sample_size: int = 2000) -> dict:
    df = pd.read_parquet(features_path)
    model, feature_cols = predict_mod.load_model(model_dir)

    # SHAP computation scales linearly in rows; cap to keep the UI snappy.
    sub = df.dropna(subset=feature_cols)
    if len(sub) > sample_size:
        sub = sub.sample(sample_size, random_state=0).sort_values("timestamp")
    X = sub[feature_cols]

    importance = global_importance(model, X, feature_cols)

    output_dir.mkdir(parents=True, exist_ok=True)
    importance_path = output_dir / "feature_importance.parquet"
    importance.to_parquet(importance_path, index=False)
    logger.info("Saved global importance to %s (top 5 below):", importance_path)
    for _, row in importance.head(5).iterrows():
        logger.info("  %-30s  %.2f", row["feature"], row["mean_abs_shap"])

    # Also explain the most recent row for the UI's "why this prediction"
    # panel.
    if not sub.empty:
        last_row = sub.iloc[[-1]]
        local = local_explanation(model, last_row[feature_cols], feature_cols)
        local["as_of"] = str(last_row["timestamp"].iloc[0])
        local_path = output_dir / "latest_explanation.json"
        local_path.write_text(json.dumps(local, indent=2))
        logger.info("Saved latest local explanation to %s", local_path)
    else:
        local = {}

    return {"importance": importance, "latest": local}


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="SHAP-based explanations for the trained model.")
    p.add_argument("--features", type=Path, default=Path("data/processed/features.parquet"))
    p.add_argument("--model-dir", type=Path, default=Path("data/models"))
    p.add_argument("--output-dir", type=Path, default=Path("data/models"))
    p.add_argument("--sample", type=int, default=2000, help="Max rows to sample for SHAP (speed cap).")
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    if not args.features.exists():
        logger.error("Missing features parquet: %s", args.features)
        return 2
    run(args.features, args.model_dir, args.output_dir, sample_size=args.sample)
    return 0


if __name__ == "__main__":
    sys.exit(main())
