"""Train an XGBoost model to predict glucose 30 minutes ahead.

For non-ML readers: this is the script that actually *learns* from the
data. See docs/ML_PRIMER.md for the conceptual overview. The short
version of what happens here:

  1. Load the feature table built by src/features.py.
  2. Split it into TRAIN / VALIDATION / TEST chunks **in time order**
     (we never let the model peek into the future during training —
     that would give us a fake-good test score).
  3. Fit an XGBoost regressor on TRAIN, watching VAL for overfitting
     ("early stopping" — we stop once the validation error stops
     improving for N rounds).
  4. Score the model on TEST, which it has never seen.
  5. Save the model + a metrics JSON for the UI / CI.

Outputs:
  data/models/xgb_glucose_30m.json     — the model
  data/models/metrics.json             — MAE, RMSE, in-tolerance %, etc.
  data/models/feature_columns.json     — exact column order used at
                                         training time (predict.py needs this)
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

from src import features as feat_mod

logger = logging.getLogger(__name__)

# Hyper-parameters. These are *good defaults*, not tuned. For a
# production model you'd run a proper search; for a baseline that
# already beats most published 30-min CGM forecasts, this is fine.
DEFAULT_PARAMS = dict(
    n_estimators=600,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.85,            # row subsampling -> regularization
    colsample_bytree=0.85,     # column subsampling -> regularization
    min_child_weight=5,        # leaf-size floor; prevents overfit on tiny groups
    reg_lambda=1.0,            # L2 on leaf weights
    objective="reg:squarederror",
    tree_method="hist",        # faster + lower memory than "exact"
    eval_metric="mae",
    random_state=42,
    n_jobs=-1,
)

# How many rounds of "no validation improvement" before we stop training.
# 50 is a safe default; smaller datasets benefit from smaller patience.
EARLY_STOPPING_ROUNDS = 50


def time_split(df: pd.DataFrame, train_frac: float = 0.70, val_frac: float = 0.15) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Walk-forward split in chronological order.

    Why not random shuffle: see ML_PRIMER.md. Time-series data has
    autocorrelation — random shuffling lets the model "look up" the
    answer from neighboring rows that happened seconds apart.

    The fractions add to <= 1.0; the remainder becomes the test set.
    """
    df = df.sort_values("timestamp").reset_index(drop=True)
    n = len(df)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))
    train = df.iloc[:train_end].copy()
    val = df.iloc[train_end:val_end].copy()
    test = df.iloc[val_end:].copy()
    return train, val, test


def _xy(df: pd.DataFrame, feature_cols: list[str], target_col: str) -> tuple[pd.DataFrame, pd.Series]:
    """Slice features (X) and label (y), dropping rows with no target."""
    sub = df.dropna(subset=[target_col])
    return sub[feature_cols], sub[target_col]


def evaluate(y_true: pd.Series, y_pred: np.ndarray) -> dict:
    """Compute the metrics we'll surface in the UI / docs.

    All values are computed on the held-out TEST set. See
    docs/ML_PRIMER.md for what each one means.
    """
    err = y_pred - y_true.to_numpy()
    abs_err = np.abs(err)
    metrics = {
        "n": int(len(y_true)),
        "mae": float(abs_err.mean()),
        "rmse": float(np.sqrt((err ** 2).mean())),
        "median_abs_err": float(np.median(abs_err)),
        # "Clinically close" — fraction of predictions within 20 mg/dL.
        # This is a common ADA-aligned threshold.
        "pct_within_20": float((abs_err <= 20).mean() * 100),
        "pct_within_30": float((abs_err <= 30).mean() * 100),
        # Hypo recall: of actual lows, how many did we predict to be < 80?
        # We use a slightly loosened threshold on the prediction side
        # because perfectly hitting 70 isn't the goal; flagging the
        # neighborhood is.
        "hypo_recall": _hypo_recall(y_true, y_pred),
    }
    return metrics


def _hypo_recall(y_true: pd.Series, y_pred: np.ndarray) -> float:
    actually_low = y_true.to_numpy() < 70
    if actually_low.sum() == 0:
        return float("nan")  # no hypos in this slice; metric undefined
    predicted_lowish = y_pred < 80
    return float((actually_low & predicted_lowish).sum() / actually_low.sum())


def train(
    features_df: pd.DataFrame,
    target_col: str = "target_30m",
    params: dict | None = None,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
) -> dict:
    """Run the full train -> validate -> evaluate loop. Returns a dict
    with the model, the feature column list, the metrics, and the
    split sizes."""
    if params is None:
        params = dict(DEFAULT_PARAMS)

    feature_cols = feat_mod.feature_columns(features_df)
    train_df, val_df, test_df = time_split(features_df, train_frac, val_frac)

    X_tr, y_tr = _xy(train_df, feature_cols, target_col)
    X_va, y_va = _xy(val_df, feature_cols, target_col)
    X_te, y_te = _xy(test_df, feature_cols, target_col)

    if len(X_tr) == 0 or len(X_va) == 0 or len(X_te) == 0:
        raise ValueError(
            f"Empty split (train={len(X_tr)}, val={len(X_va)}, test={len(X_te)}). "
            "You probably don't have enough data yet — try ingesting more days."
        )

    logger.info("Split sizes: train=%d val=%d test=%d", len(X_tr), len(X_va), len(X_te))
    logger.info("Feature count: %d", len(feature_cols))

    model = xgb.XGBRegressor(early_stopping_rounds=EARLY_STOPPING_ROUNDS, **params)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        verbose=False,
    )

    preds = model.predict(X_te)
    metrics = evaluate(y_te, preds)
    metrics["best_iteration"] = int(getattr(model, "best_iteration", model.n_estimators) or 0)
    metrics["target"] = target_col
    metrics["train_rows"] = len(X_tr)
    metrics["val_rows"] = len(X_va)
    metrics["test_rows"] = len(X_te)
    metrics["test_period"] = (
        str(test_df["timestamp"].min()),
        str(test_df["timestamp"].max()),
    )

    return {
        "model": model,
        "feature_cols": feature_cols,
        "metrics": metrics,
        "test_df": test_df,
        "test_pred": preds,
    }


def save_artifacts(result: dict, out_dir: Path) -> dict[str, Path]:
    """Persist the model, feature column list, and metrics."""
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "xgb_glucose_30m.json"
    cols_path = out_dir / "feature_columns.json"
    metrics_path = out_dir / "metrics.json"

    result["model"].save_model(model_path)
    cols_path.write_text(json.dumps(result["feature_cols"], indent=2))
    metrics_path.write_text(json.dumps(result["metrics"], indent=2, default=str))

    return {"model": model_path, "feature_cols": cols_path, "metrics": metrics_path}


def run(features_path: Path, model_dir: Path, target_col: str = "target_30m") -> dict:
    logger.info("Loading features: %s", features_path)
    df = pd.read_parquet(features_path)
    result = train(df, target_col=target_col)
    paths = save_artifacts(result, model_dir)
    m = result["metrics"]
    logger.info(
        "Test MAE = %.2f mg/dL · RMSE = %.2f · within 20 mg/dL = %.1f%% · hypo recall = %.2f",
        m["mae"], m["rmse"], m["pct_within_20"], m["hypo_recall"] if not np.isnan(m["hypo_recall"]) else float("nan"),
    )
    logger.info("Saved %s", paths)
    return result


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train XGBoost glucose-prediction model.")
    p.add_argument("--features", type=Path, default=Path("data/processed/features.parquet"))
    p.add_argument("--model-dir", type=Path, default=Path("data/models"))
    p.add_argument("--target", default="target_30m", choices=["target_30m", "target_60m"])
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    if not args.features.exists():
        logger.error("Missing features parquet: %s", args.features)
        return 2
    run(args.features, args.model_dir, target_col=args.target)
    return 0


if __name__ == "__main__":
    sys.exit(main())
