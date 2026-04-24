"""Ingest a Dexcom Clarity CSV export into a clean parquet file.

Clarity exports lump several event types into a single CSV (EGV readings,
calibrations, alerts, device events). For glucose modeling we only want
EGV (Estimated Glucose Value) rows, normalized to UTC, with trend arrows
encoded as small integers and Dexcom's "Low" / "High" sentinels coerced
to numeric clamps that match the device's reportable range (40-400 mg/dL,
with 39 / 401 used here as out-of-range markers).
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# Dexcom G7 reports glucose in the 40-400 mg/dL range. Values outside that
# clinical reportable range are exported as the literal strings "Low" /
# "High". We encode them as 39 / 401 so downstream models see a numeric
# but distinguishable signal.
LOW_SENTINEL = 39
HIGH_SENTINEL = 401

# Clarity's trend arrow strings, encoded to a compact -2..+2 integer
# scale (per the project spec). Double up/down arrows clip to +/-2 since
# rate-of-change beyond that is rare and the model can recover the
# saturated signal from glucose deltas anyway. NaN means no trend
# available (sensor warmup, missing reading) — the merge layer decides
# whether to forward-fill.
TREND_ENCODING: dict[str, int] = {
    "doubledown": -2,
    "singledown": -2,
    "fortyfivedown": -1,
    "flat": 0,
    "none": 0,  # Dexcom emits "None" for steady; treat as flat.
    "fortyfiveup": 1,
    "singleup": 2,
    "doubleup": 2,
}


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Lowercase + snake_case Clarity's verbose column names."""
    df = df.rename(columns=lambda c: c.strip().lower().replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_"))
    return df


def _find_column(df: pd.DataFrame, *candidates: str) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"None of {candidates} found in columns: {list(df.columns)}")


def _parse_glucose(value: object) -> float:
    """Coerce glucose to float, mapping Low/High strings to sentinel ints."""
    if value is None:
        return float("nan")
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value).strip()
    if not s or s.lower() in {"nan", "none"}:
        return float("nan")
    if s.lower() == "low":
        return float(LOW_SENTINEL)
    if s.lower() == "high":
        return float(HIGH_SENTINEL)
    try:
        return float(s)
    except ValueError:
        return float("nan")


def _encode_trend(value: object) -> float:
    if value is None:
        return float("nan")
    s = str(value).strip().lower().replace(" ", "")
    if not s or s in {"nan", "none"} and value != "None":
        # Treat literal NaN / empty as missing, but Dexcom's "None" string
        # is its label for "flat" — the dict above handles that.
        return float("nan")
    return float(TREND_ENCODING.get(s, float("nan")))


def load_clarity_csv(csv_path: Path) -> pd.DataFrame:
    """Read a Clarity CSV with permissive parsing (Dexcom adds metadata rows)."""
    # Clarity prepends 10+ metadata rows (patient name, device, transmitter
    # serial). The data table starts where "Index" appears in column 0.
    raw = pd.read_csv(csv_path, dtype=str, keep_default_na=False, na_values=[""])
    raw = _normalize_columns(raw)
    if "index" in raw.columns:
        # Drop pure-metadata rows that have no event type.
        event_col = _find_column(raw, "event_type")
        raw = raw[raw[event_col].notna() & (raw[event_col] != "")]
    return raw


def clean_dexcom(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to EGV rows, normalize to UTC, encode trends, drop dupes."""
    event_col = _find_column(df, "event_type")
    ts_col = _find_column(df, "timestamp_yyyy-mm-ddthh:mm:ss", "timestamp", "device_timestamp")
    glucose_col = _find_column(df, "glucose_value_mg_dl", "glucose_value_mmol_l", "glucose_value")
    trend_col = None
    for candidate in ("trend_arrow", "rate_of_change_mg_dl_min"):
        if candidate in df.columns:
            trend_col = candidate
            break

    egv_mask = df[event_col].str.upper().eq("EGV")
    egv = df.loc[egv_mask, [ts_col, glucose_col] + ([trend_col] if trend_col else [])].copy()

    # Clarity exports timestamps in the patient's local time without an
    # offset. tz_localize -> UTC keeps a single canonical timezone for
    # joins with the pump data. If the user sets DEXCOM_TZ we respect it;
    # otherwise we assume UTC (safest default — user can re-run with a tz).
    parsed_ts = pd.to_datetime(egv[ts_col], errors="coerce")
    if parsed_ts.dt.tz is None:
        parsed_ts = parsed_ts.dt.tz_localize("UTC", nonexistent="shift_forward", ambiguous="NaT")
    else:
        parsed_ts = parsed_ts.dt.tz_convert("UTC")

    out = pd.DataFrame(
        {
            "timestamp": parsed_ts,
            "glucose_mg_dl": egv[glucose_col].map(_parse_glucose),
        }
    )
    if trend_col == "trend_arrow":
        out["trend_arrow_encoded"] = egv[trend_col].map(_encode_trend)
    elif trend_col == "rate_of_change_mg_dl_min":
        # Convert numeric rate-of-change (mg/dL/min) to the same -2..+2
        # scale so downstream code only sees one trend representation.
        roc = pd.to_numeric(egv[trend_col], errors="coerce")
        out["trend_arrow_encoded"] = roc.clip(-2, 2).round().astype("Int8").astype("float")
    else:
        out["trend_arrow_encoded"] = float("nan")

    # Drop calibration-only / null glucose rows. A NaN glucose with a
    # valid timestamp typically means a calibration entry that slipped
    # past the EGV filter on older Clarity exports.
    out = out.dropna(subset=["timestamp", "glucose_mg_dl"])

    # Duplicate timestamps occur when a sensor restart re-emits a reading.
    # Keep the last one — it reflects the most recent calibration state.
    out = out.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last")

    out["trend_arrow_encoded"] = out["trend_arrow_encoded"].astype("float32")
    out["glucose_mg_dl"] = out["glucose_mg_dl"].astype("float32")
    return out.reset_index(drop=True)


def ingest(csv_path: Path, output_path: Path) -> pd.DataFrame:
    logger.info("Loading Clarity CSV: %s", csv_path)
    raw = load_clarity_csv(csv_path)
    logger.info("Raw rows: %d", len(raw))
    cleaned = clean_dexcom(raw)
    logger.info("Clean EGV rows: %d (range %s -> %s)", len(cleaned), cleaned["timestamp"].min(), cleaned["timestamp"].max())
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_parquet(output_path, index=False)
    logger.info("Wrote %s", output_path)
    return cleaned


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Clean a Dexcom Clarity CSV export.")
    p.add_argument("csv", type=Path, help="Path to Clarity CSV export")
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("data/processed/dexcom_clean.parquet"),
        help="Output parquet path",
    )
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    if not args.csv.exists():
        logger.error("CSV not found: %s", args.csv)
        return 2
    ingest(args.csv, args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
