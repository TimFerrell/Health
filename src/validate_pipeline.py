"""Validate the unified glucose+insulin timeline and produce EDA plots.

Outputs (under --plot-dir):
  glucose_with_boluses.html   — interactive plotly chart with bolus pins
  iob_overlay.png             — matplotlib chart of IOB vs glucose

Stdout:
  shape, date range, null counts, gap report, time-in-range stats.

Time-in-range bands (ADA pediatric T1D consensus):
  < 70  mg/dL : low / hypo
  70-180     : in range
  > 180      : high / hyper
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless-safe; we only ever save to disk
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

GAP_THRESHOLD = pd.Timedelta(minutes=30)


def summarize(df: pd.DataFrame) -> dict:
    info = {
        "shape": df.shape,
        "date_range": (df["timestamp"].min(), df["timestamp"].max()),
        "null_counts": df.isna().sum().to_dict(),
    }
    return info


def find_gaps(df: pd.DataFrame, threshold: pd.Timedelta = GAP_THRESHOLD) -> pd.DataFrame:
    """Return start/end/duration of runs with no glucose readings."""
    g = df.dropna(subset=["glucose_mg_dl"]).sort_values("timestamp")
    if g.empty:
        return pd.DataFrame(columns=["gap_start", "gap_end", "duration_minutes"])
    deltas = g["timestamp"].diff()
    mask = deltas > threshold
    gaps = pd.DataFrame(
        {
            "gap_start": g["timestamp"].shift(1)[mask].values,
            "gap_end": g["timestamp"][mask].values,
        }
    )
    gaps["duration_minutes"] = (gaps["gap_end"] - gaps["gap_start"]).dt.total_seconds() / 60.0
    return gaps


def time_in_range(df: pd.DataFrame) -> dict:
    g = df["glucose_mg_dl"].dropna()
    if g.empty:
        return {"low_pct": 0.0, "in_range_pct": 0.0, "high_pct": 0.0, "n_readings": 0}
    n = len(g)
    return {
        "low_pct": float((g < 70).sum() / n * 100),
        "in_range_pct": float(((g >= 70) & (g <= 180)).sum() / n * 100),
        "high_pct": float((g > 180).sum() / n * 100),
        "n_readings": int(n),
    }


def plot_glucose_with_boluses(df: pd.DataFrame, out_path: Path) -> None:
    """Interactive plotly chart: glucose line + bolus event pins."""
    try:
        import plotly.graph_objects as go
    except ImportError:
        logger.warning("plotly not installed; skipping interactive plot")
        return

    boluses = df[(df["bolus_units"].fillna(0) > 0)]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["timestamp"],
            y=df["glucose_mg_dl"],
            mode="lines",
            name="Glucose (mg/dL)",
            line=dict(width=1.2),
        )
    )
    # Target band shading
    fig.add_hrect(y0=70, y1=180, fillcolor="green", opacity=0.08, line_width=0)
    if not boluses.empty:
        fig.add_trace(
            go.Scatter(
                x=boluses["timestamp"],
                y=[40] * len(boluses),  # park markers along the bottom axis
                mode="markers",
                name="Bolus",
                marker=dict(symbol="triangle-up", size=8, color="orange"),
                text=[f"{u:.2f} U" for u in boluses["bolus_units"]],
                hovertemplate="%{x}<br>%{text}<extra>bolus</extra>",
            )
        )
    fig.update_layout(
        title="Glucose timeline with bolus events",
        xaxis_title="Time (UTC)",
        yaxis_title="Glucose (mg/dL)",
        yaxis=dict(range=[30, 410]),
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(out_path)
    logger.info("Wrote %s", out_path)


def plot_iob_overlay(df: pd.DataFrame, out_path: Path) -> None:
    """Two-axis matplotlib chart: glucose (left) + IOB (right)."""
    fig, ax1 = plt.subplots(figsize=(14, 5))
    ax1.plot(df["timestamp"], df["glucose_mg_dl"], color="#1f77b4", linewidth=1.0, label="Glucose")
    ax1.set_ylabel("Glucose (mg/dL)", color="#1f77b4")
    ax1.axhspan(70, 180, alpha=0.08, color="green")
    ax1.set_ylim(30, 410)

    ax2 = ax1.twinx()
    ax2.plot(df["timestamp"], df["iob_units"], color="#ff7f0e", linewidth=1.0, alpha=0.8, label="IOB")
    ax2.set_ylabel("IOB (units)", color="#ff7f0e")
    ax2.set_ylim(bottom=0)

    ax1.set_title("Glucose vs Insulin On Board (IOB decay sanity check)")
    fig.autofmt_xdate()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    logger.info("Wrote %s", out_path)


def report(df: pd.DataFrame, plot_dir: Path) -> None:
    info = summarize(df)
    print("=" * 72)
    print("UNIFIED TIMELINE — SHAPE & RANGE")
    print("=" * 72)
    print(f"shape:       {info['shape']}")
    print(f"date range:  {info['date_range'][0]}  ->  {info['date_range'][1]}")
    print()
    print("NULL COUNTS BY COLUMN")
    for k, v in info["null_counts"].items():
        print(f"  {k:25s} {v}")
    print()

    gaps = find_gaps(df)
    print(f"GLUCOSE GAPS > {int(GAP_THRESHOLD.total_seconds() / 60)} MINUTES: {len(gaps)}")
    if not gaps.empty:
        for _, g in gaps.head(10).iterrows():
            print(f"  {g['gap_start']}  ->  {g['gap_end']}   ({g['duration_minutes']:.0f} min)")
        if len(gaps) > 10:
            print(f"  ... ({len(gaps) - 10} more)")
    print()

    tir = time_in_range(df)
    print("TIME IN RANGE")
    print(f"  readings:     {tir['n_readings']}")
    print(f"  < 70   (low): {tir['low_pct']:.1f}%")
    print(f"  70-180 (TIR): {tir['in_range_pct']:.1f}%")
    print(f"  > 180  (hi):  {tir['high_pct']:.1f}%")
    print()

    plot_dir.mkdir(parents=True, exist_ok=True)
    plot_glucose_with_boluses(df, plot_dir / "glucose_with_boluses.html")
    plot_iob_overlay(df, plot_dir / "iob_overlay.png")


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Validate and plot the unified glucose+insulin timeline.")
    p.add_argument("--input", type=Path, default=Path("data/processed/unified_timeline.parquet"))
    p.add_argument("--plot-dir", type=Path, default=Path("plots"))
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    if not args.input.exists():
        logger.error("Missing unified parquet: %s", args.input)
        return 2
    df = pd.read_parquet(args.input)
    report(df, args.plot_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
