"""Light Streamlit UI for the T1D glucose pipeline.

Sections (sidebar nav):
  Pipeline  — upload a Clarity CSV, choose a Tandem date range, run each
              phase, see file/status output.
  Timeline  — interactive glucose chart with bolus pins, IOB overlay,
              and date-range filter.
  Stats     — time-in-range, gap detection, null counts.

Run with:
    streamlit run app.py
"""
from __future__ import annotations

import io
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

from src import ingest_dexcom, ingest_tandem, merge_pipeline, validate_pipeline  # noqa: E402

DATA_DIR = REPO / "data" / "processed"
RAW_DIR = REPO / "data" / "raw"
DEXCOM_PARQUET = DATA_DIR / "dexcom_clean.parquet"
TANDEM_PARQUET = DATA_DIR / "tandem_clean.parquet"
UNIFIED_PARQUET = DATA_DIR / "unified_timeline.parquet"

st.set_page_config(page_title="T1D Glucose Pipeline", page_icon="🩸", layout="wide")


# ---------- helpers ----------------------------------------------------------

def _file_status(path: Path) -> str:
    if not path.exists():
        return "missing"
    mtime = datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
    size_kb = path.stat().st_size / 1024
    return f"{size_kb:,.1f} KB · {mtime}"


@st.cache_data(show_spinner=False)
def _load_parquet(path: str, mtime: float) -> pd.DataFrame:
    """Cache-keyed on path + mtime so re-running a phase invalidates."""
    del mtime  # only used as cache key
    return pd.read_parquet(path)


def _load_unified() -> pd.DataFrame | None:
    if not UNIFIED_PARQUET.exists():
        return None
    return _load_parquet(str(UNIFIED_PARQUET), UNIFIED_PARQUET.stat().st_mtime)


# ---------- sidebar ----------------------------------------------------------

st.sidebar.title("🩸 T1D Pipeline")
section = st.sidebar.radio("Section", ["Pipeline", "Timeline", "Stats"], label_visibility="collapsed")

st.sidebar.divider()
st.sidebar.caption("Artifacts")
st.sidebar.write(f"**dexcom**: `{_file_status(DEXCOM_PARQUET)}`")
st.sidebar.write(f"**tandem**: `{_file_status(TANDEM_PARQUET)}`")
st.sidebar.write(f"**unified**: `{_file_status(UNIFIED_PARQUET)}`")


# ---------- Pipeline section -------------------------------------------------

if section == "Pipeline":
    st.title("Pipeline")
    st.caption("Run each phase independently. Outputs land in `data/processed/`.")

    # --- Phase 1 ---------------------------------------------------------
    st.subheader("Phase 1 — Dexcom Clarity ingestion")
    with st.container(border=True):
        uploaded = st.file_uploader("Clarity CSV export", type=["csv"], key="dexcom_csv")
        col_a, col_b = st.columns([1, 3])
        with col_a:
            run_dexcom = st.button("Run Phase 1", type="primary", disabled=uploaded is None)
        with col_b:
            st.write(f"Output → `{DEXCOM_PARQUET.relative_to(REPO)}`")

        if run_dexcom and uploaded is not None:
            RAW_DIR.mkdir(parents=True, exist_ok=True)
            tmp_csv = RAW_DIR / uploaded.name
            tmp_csv.write_bytes(uploaded.getvalue())
            with st.spinner("Cleaning Clarity CSV…"):
                try:
                    df = ingest_dexcom.ingest(tmp_csv, DEXCOM_PARQUET)
                    st.success(f"Wrote {len(df):,} EGV rows.")
                    st.dataframe(df.head(20), width="stretch", hide_index=True)
                except Exception as e:  # noqa: BLE001
                    st.error(f"Ingestion failed: {e}")

    # --- Phase 2 ---------------------------------------------------------
    st.subheader("Phase 2 — Tandem pump ingestion")
    with st.container(border=True):
        col1, col2, col3 = st.columns(3)
        default_end = datetime.now(tz=timezone.utc).date()
        default_start = default_end - timedelta(days=30)
        start = col1.date_input("Start date", value=default_start, key="tandem_start")
        end = col2.date_input("End date", value=default_end, key="tandem_end")
        allow_synth = col3.toggle("Allow synthetic fallback", value=True,
                                  help="If t:connect auth fails, generate realistic synthetic pump data.")

        run_tandem = st.button("Run Phase 2", type="primary")
        st.write(f"Output → `{TANDEM_PARQUET.relative_to(REPO)}`")

        if run_tandem:
            cfg = ingest_tandem.TandemConfig.from_env(
                start=datetime.combine(start, datetime.min.time(), tzinfo=timezone.utc),
                end=datetime.combine(end, datetime.max.time(), tzinfo=timezone.utc),
            )
            with st.spinner("Pulling pump events…"):
                try:
                    df = ingest_tandem.ingest(cfg, TANDEM_PARQUET, allow_synthetic=allow_synth)
                    st.success(f"Wrote {len(df):,} events.")
                    counts = df["event_type"].value_counts().rename_axis("event_type").reset_index(name="count")
                    st.dataframe(counts, width="stretch", hide_index=True)
                except Exception as e:  # noqa: BLE001
                    st.error(f"Ingestion failed: {e}")

    # --- Phase 3 ---------------------------------------------------------
    st.subheader("Phase 3 — Merge to 5-min grid")
    with st.container(border=True):
        run_merge = st.button(
            "Run Phase 3",
            type="primary",
            disabled=not DEXCOM_PARQUET.exists(),
            help=None if DEXCOM_PARQUET.exists() else "Run Phase 1 first",
        )
        st.write(f"Output → `{UNIFIED_PARQUET.relative_to(REPO)}`")

        if run_merge:
            with st.spinner("Merging…"):
                try:
                    df = merge_pipeline.run(DEXCOM_PARQUET, TANDEM_PARQUET, UNIFIED_PARQUET)
                    st.success(f"Wrote {len(df):,} rows.")
                    st.dataframe(df.head(20), width="stretch", hide_index=True)
                except Exception as e:  # noqa: BLE001
                    st.error(f"Merge failed: {e}")


# ---------- Timeline section -------------------------------------------------

elif section == "Timeline":
    st.title("Timeline")
    df = _load_unified()
    if df is None:
        st.info("No unified timeline yet. Run Phase 3 first.")
        st.stop()

    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    min_d = df["timestamp"].min().date()
    max_d = df["timestamp"].max().date()
    col_a, col_b = st.columns(2)
    start = col_a.date_input("From", value=max(min_d, max_d - timedelta(days=3)), min_value=min_d, max_value=max_d)
    end = col_b.date_input("To", value=max_d, min_value=min_d, max_value=max_d)

    mask = (df["timestamp"].dt.date >= start) & (df["timestamp"].dt.date <= end)
    view = df.loc[mask].copy()
    if view.empty:
        st.warning("No rows in selected window.")
        st.stop()

    # --- Glucose + bolus chart -------------------------------------------
    fig = go.Figure()
    fig.add_hrect(y0=70, y1=180, fillcolor="green", opacity=0.08, line_width=0)
    fig.add_trace(go.Scatter(
        x=view["timestamp"], y=view["glucose_mg_dl"],
        mode="lines", name="Glucose", line=dict(width=1.4, color="#1f77b4"),
    ))
    boluses = view[view["bolus_units"].fillna(0) > 0]
    if not boluses.empty:
        fig.add_trace(go.Scatter(
            x=boluses["timestamp"], y=[45] * len(boluses),
            mode="markers", name="Bolus",
            marker=dict(symbol="triangle-up", size=10, color="orange"),
            text=[f"{u:.2f} U" for u in boluses["bolus_units"]],
            hovertemplate="%{x}<br>%{text}<extra>bolus</extra>",
        ))
    suspends = view[view["is_suspended"]]
    if not suspends.empty:
        fig.add_trace(go.Scatter(
            x=suspends["timestamp"], y=[400] * len(suspends),
            mode="markers", name="CIQ Suspend",
            marker=dict(symbol="square", size=6, color="red"),
        ))
    fig.update_layout(
        title="Glucose · bolus · suspend",
        yaxis=dict(title="mg/dL", range=[30, 410]),
        xaxis=dict(title="UTC"),
        height=420, legend=dict(orientation="h"),
    )
    st.plotly_chart(fig, width="stretch")

    # --- IOB overlay -----------------------------------------------------
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=view["timestamp"], y=view["glucose_mg_dl"],
        mode="lines", name="Glucose", yaxis="y", line=dict(color="#1f77b4", width=1.0),
    ))
    fig2.add_trace(go.Scatter(
        x=view["timestamp"], y=view["iob_units"],
        mode="lines", name="IOB", yaxis="y2", line=dict(color="#ff7f0e", width=1.2),
        fill="tozeroy", fillcolor="rgba(255,127,14,0.15)",
    ))
    fig2.update_layout(
        title="IOB decay vs glucose",
        height=320, legend=dict(orientation="h"),
        yaxis=dict(title="Glucose (mg/dL)"),
        yaxis2=dict(title="IOB (U)", overlaying="y", side="right", showgrid=False),
    )
    st.plotly_chart(fig2, width="stretch")

    with st.expander("Raw rows in window"):
        st.dataframe(view, width="stretch", hide_index=True)


# ---------- Stats section ----------------------------------------------------

else:  # Stats
    st.title("Stats")
    df = _load_unified()
    if df is None:
        st.info("No unified timeline yet. Run Phase 3 first.")
        st.stop()

    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    tir = validate_pipeline.time_in_range(df)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Readings", f"{tir['n_readings']:,}")
    c2.metric("Low <70", f"{tir['low_pct']:.1f}%")
    c3.metric("In range 70-180", f"{tir['in_range_pct']:.1f}%")
    c4.metric("High >180", f"{tir['high_pct']:.1f}%")

    st.subheader("Date range")
    st.write(f"{df['timestamp'].min()} → {df['timestamp'].max()}  ({len(df):,} rows on a 5-min grid)")

    st.subheader("Null counts")
    nulls = df.isna().sum().rename("nulls").to_frame()
    nulls["non_null"] = len(df) - nulls["nulls"]
    st.dataframe(nulls, width="stretch")

    st.subheader("Glucose gaps > 30 min")
    gaps = validate_pipeline.find_gaps(df)
    if gaps.empty:
        st.success("No gaps detected.")
    else:
        st.warning(f"{len(gaps)} gap(s) detected.")
        st.dataframe(gaps, width="stretch", hide_index=True)

    st.subheader("Pump activity summary")
    summary = pd.DataFrame({
        "metric": ["Total bolus units", "Peak IOB (U)", "Suspend bins", "CIQ-active bins"],
        "value": [
            f"{df['bolus_units'].sum():.1f}",
            f"{df['iob_units'].max():.2f}",
            f"{int(df['is_suspended'].sum())}",
            f"{int(df['control_iq_active'].sum())}",
        ],
    })
    st.dataframe(summary, width="stretch", hide_index=True)

    csv_buf = io.BytesIO()
    df.to_csv(csv_buf, index=False)
    st.download_button(
        "Download unified timeline (CSV)",
        data=csv_buf.getvalue(),
        file_name="unified_timeline.csv",
        mime="text/csv",
    )
