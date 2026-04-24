"""Merge cleaned Dexcom and Tandem parquets onto a single 5-minute grid.

Why 5 minutes:
  Dexcom G7 reports a new EGV every ~5 minutes. Aligning pump events to
  that cadence keeps the model's input granularity matched to the
  ground-truth signal it has to predict.

Why exponential IOB decay:
  Rapid-acting analogs (Humalog, Novolog, Fiasp) have a one-compartment
  PK profile that's well approximated by IOB(t) = dose * exp(-t/tau)
  with tau ~= 240 minutes (4 hr half-life is the clinical convention,
  even though peak action is ~75 min). When the pump's own IOB stream
  is missing, this back-of-envelope decay gives the model a usable
  insulin-on-board feature without needing a full PK model yet.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

GRID_FREQ = "5min"
IOB_HALFLIFE_MIN = 240.0  # 4 hours; clinical convention for rapid-acting analogs
# Decay rate so IOB(t) = dose * exp(-t / IOB_TAU). Using tau = halflife as
# spec'd (the spec literally says e^(-t/240); a "true" half-life decay
# would divide by halflife/ln(2), but we honor the spec verbatim).
IOB_TAU_MIN = IOB_HALFLIFE_MIN
# How far forward to project a bolus's IOB contribution (5 half-lives
# is < 1% of the original dose, safe to truncate).
IOB_HORIZON_MIN = int(IOB_HALFLIFE_MIN * 5)


def _build_grid(start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
    start = start.floor(GRID_FREQ)
    end = end.ceil(GRID_FREQ)
    return pd.date_range(start, end, freq=GRID_FREQ, tz="UTC")


def resample_glucose(dexcom: pd.DataFrame, grid: pd.DatetimeIndex) -> pd.DataFrame:
    """Snap CGM readings to the 5-min grid using nearest-neighbor within
    a 5-minute tolerance. Wider tolerances would smear sensor warmup
    artifacts across legitimate gaps."""
    dx = dexcom.set_index("timestamp").sort_index()
    # Resample to 5-min bins, taking the mean within each bin (G7 can
    # occasionally emit two readings inside a bin after a sensor
    # restart). Then reindex to the canonical grid.
    binned = dx.resample(GRID_FREQ).mean(numeric_only=True)
    binned = binned.reindex(grid)
    binned.index.name = "timestamp"
    return binned.reset_index()


def _expand_bolus_iob(boluses: pd.DataFrame, grid: pd.DatetimeIndex) -> pd.Series:
    """Project each bolus's IOB contribution onto the grid via exponential decay.

    Returns a Series indexed by grid timestamps with summed IOB units.
    """
    if boluses.empty:
        return pd.Series(0.0, index=grid, name="iob_units_modeled")

    grid_minutes = (grid.view("int64") // 60_000_000_000).astype(np.int64)
    iob = np.zeros(len(grid), dtype=np.float64)

    bolus_ts = (boluses["timestamp"].view("int64") // 60_000_000_000).to_numpy()
    bolus_doses = boluses["value"].fillna(0.0).to_numpy()

    for ts_min, dose in zip(bolus_ts, bolus_doses):
        if dose <= 0:
            continue
        # Find the first grid index at or after the bolus
        start_idx = int(np.searchsorted(grid_minutes, ts_min, side="left"))
        end_idx = int(np.searchsorted(grid_minutes, ts_min + IOB_HORIZON_MIN, side="right"))
        if start_idx >= len(grid):
            continue
        elapsed = grid_minutes[start_idx:end_idx] - ts_min
        iob[start_idx:end_idx] += dose * np.exp(-elapsed / IOB_TAU_MIN)

    return pd.Series(iob, index=grid, name="iob_units_modeled")


def _expand_basal(basal: pd.DataFrame, grid: pd.DatetimeIndex) -> pd.Series:
    """Forward-fill basal rate samples onto the 5-min grid."""
    if basal.empty:
        return pd.Series(np.nan, index=grid, name="basal_rate")
    s = basal.set_index("timestamp")["value"].sort_index()
    s = s[~s.index.duplicated(keep="last")]
    return s.reindex(grid, method="ffill").rename("basal_rate")


def _expand_boluses_per_bin(boluses: pd.DataFrame, grid: pd.DatetimeIndex) -> pd.Series:
    """Sum bolus units delivered within each 5-min bin (the actual event
    column, distinct from the projected IOB curve)."""
    if boluses.empty:
        return pd.Series(0.0, index=grid, name="bolus_units")
    s = boluses.set_index("timestamp")["value"].sort_index().fillna(0.0)
    binned = s.resample(GRID_FREQ).sum().reindex(grid, fill_value=0.0)
    return binned.rename("bolus_units")


def _expand_suspends(events: pd.DataFrame, grid: pd.DatetimeIndex) -> pd.Series:
    """Mark grid bins that fall inside an active suspend window."""
    flag = pd.Series(False, index=grid, name="is_suspended")
    if events.empty:
        return flag
    for _, row in events.iterrows():
        start = row["timestamp"]
        dur = row.get("duration_minutes") or 0.0
        end = start + pd.Timedelta(minutes=float(dur if dur > 0 else 5.0))
        flag.loc[(grid >= start) & (grid < end)] = True
    return flag


def _expand_ciq_actions(events: pd.DataFrame, grid: pd.DatetimeIndex) -> pd.Series:
    """Mark grid bins where Control-IQ took an algorithmic action.

    Useful as a "the loop is reacting to something" feature, even when
    we can't fully decode what the algorithm was responding to.
    """
    flag = pd.Series(False, index=grid, name="control_iq_active")
    if events.empty:
        return flag
    for _, row in events.iterrows():
        start = row["timestamp"]
        dur = row.get("duration_minutes") or 5.0
        end = start + pd.Timedelta(minutes=float(dur if dur > 0 else 5.0))
        flag.loc[(grid >= start) & (grid < end)] = True
    return flag


def merge(dexcom: pd.DataFrame, tandem: pd.DataFrame) -> pd.DataFrame:
    if dexcom.empty:
        raise ValueError("dexcom dataframe is empty; cannot build a glucose timeline")

    grid_start = min(dexcom["timestamp"].min(), tandem["timestamp"].min() if not tandem.empty else dexcom["timestamp"].min())
    grid_end = max(dexcom["timestamp"].max(), tandem["timestamp"].max() if not tandem.empty else dexcom["timestamp"].max())
    grid = _build_grid(grid_start, grid_end)

    glucose = resample_glucose(dexcom, grid).set_index("timestamp")

    boluses = tandem[tandem["event_type"] == "bolus"].copy() if not tandem.empty else tandem
    basal = tandem[tandem["event_type"] == "basal"].copy() if not tandem.empty else tandem
    suspends = tandem[tandem["event_type"] == "suspend"].copy() if not tandem.empty else tandem
    ciq = tandem[tandem["event_type"] == "control_iq_action"].copy() if not tandem.empty else tandem

    bolus_per_bin = _expand_boluses_per_bin(boluses, grid)
    basal_series = _expand_basal(basal, grid)
    suspend_flag = _expand_suspends(suspends, grid)
    ciq_flag = _expand_ciq_actions(ciq, grid)

    # Prefer the pump's own IOB if it exists for any bolus; otherwise
    # fall back to the modeled exponential decay. We resample the pump's
    # IOB stream by taking the max within a bin (IOB doesn't sum across
    # boluses if it's already a stock value) — but if the column is all
    # NaN we skip it entirely.
    has_real_iob = not boluses.empty and not boluses["iob_units"].isna().all()
    if has_real_iob:
        real_iob = (
            boluses.set_index("timestamp")["iob_units"]
            .sort_index()
            .resample(GRID_FREQ)
            .max()
            .reindex(grid)
            .ffill()
            .fillna(0.0)
        )
        iob_series = real_iob.rename("iob_units")
    else:
        iob_series = _expand_bolus_iob(boluses, grid).rename("iob_units")

    out = glucose.join(
        [
            iob_series,
            bolus_per_bin,
            basal_series,
            suspend_flag,
            ciq_flag,
        ]
    )
    out = out.reset_index()

    # Final schema + dtypes
    schema_cols = [
        "timestamp",
        "glucose_mg_dl",
        "trend_arrow_encoded",
        "iob_units",
        "bolus_units",
        "basal_rate",
        "is_suspended",
        "control_iq_active",
    ]
    for c in schema_cols:
        if c not in out.columns:
            out[c] = np.nan
    out = out[schema_cols]

    out["glucose_mg_dl"] = out["glucose_mg_dl"].astype("float32")
    out["trend_arrow_encoded"] = out["trend_arrow_encoded"].astype("float32")
    out["iob_units"] = out["iob_units"].astype("float32")
    out["bolus_units"] = out["bolus_units"].astype("float32")
    out["basal_rate"] = out["basal_rate"].astype("float32")
    out["is_suspended"] = out["is_suspended"].fillna(False).astype(bool)
    out["control_iq_active"] = out["control_iq_active"].fillna(False).astype(bool)

    return out


def run(dexcom_path: Path, tandem_path: Path, output_path: Path) -> pd.DataFrame:
    logger.info("Loading dexcom=%s tandem=%s", dexcom_path, tandem_path)
    dexcom = pd.read_parquet(dexcom_path)
    tandem = pd.read_parquet(tandem_path) if tandem_path.exists() else pd.DataFrame(
        columns=["timestamp", "event_type", "value", "duration_minutes", "iob_units"]
    )
    merged = merge(dexcom, tandem)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(output_path, index=False)
    logger.info("Wrote unified timeline (%d rows) to %s", len(merged), output_path)
    return merged


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Merge Dexcom + Tandem parquets onto a 5-min grid.")
    p.add_argument("--dexcom", type=Path, default=Path("data/processed/dexcom_clean.parquet"))
    p.add_argument("--tandem", type=Path, default=Path("data/processed/tandem_clean.parquet"))
    p.add_argument("-o", "--output", type=Path, default=Path("data/processed/unified_timeline.parquet"))
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    if not args.dexcom.exists():
        logger.error("Missing dexcom parquet: %s", args.dexcom)
        return 2
    run(args.dexcom, args.tandem, args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
