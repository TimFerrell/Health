"""Light Streamlit UI for the T1D glucose pipeline.

Sections (sidebar nav):
  Pipeline  — upload a Clarity CSV, choose a Tandem date range, run each
              phase, see file/status output.
  Timeline  — interactive glucose chart with bolus pins, IOB overlay,
              and date-range filter.
  Stats     — time-in-range, gap detection, null counts.
  Model     — train, evaluate, and explain the XGBoost predictor.
  Predict   — current 30-min forecast + backtest predicted-vs-actual.

Run with:
    streamlit run app.py
"""
from __future__ import annotations

import io
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

from src import (  # noqa: E402
    anomaly,
    counterfactual,
    drift,
    explain as explain_mod,
    features as feat_mod,
    ingest_dexcom,
    ingest_tandem,
    merge_pipeline,
    predict as predict_mod,
    refresh as refresh_mod,
    train as train_mod,
    treatments as treatments_mod,
    validate_pipeline,
)

DATA_DIR = REPO / "data" / "processed"
RAW_DIR = REPO / "data" / "raw"
MODEL_DIR = REPO / "data" / "models"
DEXCOM_PARQUET = DATA_DIR / "dexcom_clean.parquet"
TANDEM_PARQUET = DATA_DIR / "tandem_clean.parquet"
UNIFIED_PARQUET = DATA_DIR / "unified_timeline.parquet"
FEATURES_PARQUET = DATA_DIR / "features.parquet"
TREATMENTS_PARQUET = DATA_DIR / "treatments.parquet"
MODEL_PATH = MODEL_DIR / "xgb_glucose_30m.json"
METRICS_PATH = MODEL_DIR / "metrics.json"
IMPORTANCE_PATH = MODEL_DIR / "feature_importance.parquet"
LOCAL_EXPL_PATH = MODEL_DIR / "latest_explanation.json"
PREDICTIONS_LOG = DATA_DIR / "predictions_log.parquet"
ALERTS_LOG = DATA_DIR / "alerts_log.parquet"

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


def _load(path: Path) -> pd.DataFrame | None:
    """Cached load for any of our parquets. Returns None if missing."""
    if not path.exists():
        return None
    return _load_parquet(str(path), path.stat().st_mtime)


def _load_unified() -> pd.DataFrame | None:
    return _load(UNIFIED_PARQUET)


@st.cache_resource(show_spinner=False)
def _cached_model(model_dir: str, mtime: float):
    """Cache the deserialized XGBoost booster across reruns. The mtime
    key invalidates when you retrain."""
    del mtime
    return predict_mod.load_model(Path(model_dir))


def _load_model():
    if not MODEL_PATH.exists():
        return None, None
    return _cached_model(str(MODEL_DIR), MODEL_PATH.stat().st_mtime)


@st.cache_data(show_spinner=False)
def _cached_backtest(features_path: str, model_dir: str, feats_mtime: float, model_mtime: float) -> pd.DataFrame:
    """The backtest is deterministic in (features, model) — cache on
    their mtimes. ~25k row predict() is fast (~50 ms) but the
    DataFrame copy that follows is what stalls Streamlit reruns."""
    del feats_mtime, model_mtime
    feats = pd.read_parquet(features_path)
    model, cols = predict_mod.load_model(Path(model_dir))
    return predict_mod.predict_dataframe(model, cols, feats)


def _decimate(df: pd.DataFrame, n: int = 2000) -> pd.DataFrame:
    """Even-stride downsample for plotting. The eye can't resolve more
    than ~2,000 points in a glucose chart, and plotly + iOS Safari
    chokes on full 25k-point series."""
    if len(df) <= n:
        return df
    step = max(1, len(df) // n)
    return df.iloc[::step]


# ---------- sidebar ----------------------------------------------------------

st.sidebar.title("🩸 T1D Pipeline")
section = st.sidebar.radio(
    "Section",
    ["Live", "Log carbs", "Pipeline", "Timeline", "Stats", "Model", "Predict", "What-if"],
    label_visibility="collapsed",
)

st.sidebar.divider()
st.sidebar.caption("Artifacts")
st.sidebar.write(f"**dexcom**: `{_file_status(DEXCOM_PARQUET)}`")
st.sidebar.write(f"**tandem**: `{_file_status(TANDEM_PARQUET)}`")
st.sidebar.write(f"**unified**: `{_file_status(UNIFIED_PARQUET)}`")
st.sidebar.write(f"**features**: `{_file_status(FEATURES_PARQUET)}`")
st.sidebar.write(f"**model**: `{_file_status(MODEL_PATH)}`")

st.sidebar.divider()
st.sidebar.caption("[ML primer for non-ML readers](docs/ML_PRIMER.md)")


# ---------- Live section -----------------------------------------------------

if section == "Live":
    st.title("Live")
    st.caption(
        "Real-time view: current glucose, the model's 30-minute forecast, "
        "active alerts, and how the model is doing against reality."
    )

    col_a, col_b, col_c = st.columns([1, 1, 3])
    with col_a:
        run_now = st.button("Refresh now", type="primary",
                            help="Pulls the last 24 h of CGM, re-merges, re-features, predicts.")
    with col_b:
        auto_refresh_min = st.selectbox("Auto-refresh", [0, 1, 5, 10], index=0,
                                         help="Set 0 to disable. Re-runs only the live tile, not the whole app.")
    with col_c:
        if not MODEL_PATH.exists():
            st.warning("No trained model yet — go to **Model** and train first.")

    if run_now:
        with st.spinner("Pulling Dexcom Share + pump events, predicting…"):
            try:
                result = refresh_mod.cycle()
                st.success(refresh_mod._summarize(result))
            except Exception as e:  # noqa: BLE001
                st.error(f"Refresh failed: {e}")

    # --- The live tile lives in a fragment so its `run_every` rerun
    # only re-renders this block, not the whole script (no parquet
    # reloads, no heavy imports). ----------------------------------------
    fragment_interval = f"{auto_refresh_min}min" if auto_refresh_min else None

    @st.fragment(run_every=fragment_interval)
    def _live_panel():
        if not PREDICTIONS_LOG.exists():
            st.info("No predictions logged yet. Click **Refresh now** above to make the first one.")
            return

        log = _load(PREDICTIONS_LOG).sort_values("predicted_at")
        latest = log.iloc[-1]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Current", f"{latest['current_glucose']:.0f} mg/dL")
        delta = latest["prediction_30m"] - latest["current_glucose"]
        c2.metric("Predicted (+30m)", f"{latest['prediction_30m']:.0f} mg/dL", delta=f"{delta:+.0f}")
        c3.metric("As of (UTC)", str(latest["predicted_at"]).split("+")[0].replace("T", " "))
        alerts = anomaly.detect(
            current_glucose=float(latest["current_glucose"]),
            current_timestamp=latest["predicted_at"],
            prediction_30m=float(latest["prediction_30m"]),
        )
        c4.metric("Active alerts", len(alerts))

        if alerts:
            for a in alerts:
                if a.severity == "critical":
                    st.error(f"**{a.type}** — {a.message}")
                elif a.severity == "warn":
                    st.warning(f"**{a.type}** — {a.message}")
                else:
                    st.info(f"**{a.type}** — {a.message}")
        else:
            st.success("No active alerts.")

        # --- Recent predictions chart ---------------------------------------
        st.subheader("Recent predictions vs. realized glucose")
        unified = _load_unified() if UNIFIED_PARQUET.exists() else pd.DataFrame()
        scored = drift.join_with_actuals(log, unified)
        recent = _decimate(scored.tail(288))  # last 24h at 5-min cadence -> already 288 max
        fig = go.Figure()
        fig.add_hrect(y0=70, y1=180, fillcolor="green", opacity=0.06, line_width=0)
        fig.add_trace(go.Scatter(
            x=recent["target_timestamp"], y=recent["actual_30m"],
            name="Actual", line=dict(color="#1f77b4", width=1.4),
        ))
        fig.add_trace(go.Scatter(
            x=recent["target_timestamp"], y=recent["prediction_30m"],
            name="Predicted (made 30m earlier)", line=dict(color="#d62728", width=1.0, dash="dot"),
        ))
        fig.update_layout(height=380, yaxis_title="mg/dL", xaxis_title="UTC",
                          legend=dict(orientation="h"))
        st.plotly_chart(fig, width="stretch")

        # --- Drift summary --------------------------------------------------
        st.subheader("Live accuracy / drift")
        status = drift.compute_status(PREDICTIONS_LOG, UNIFIED_PARQUET, METRICS_PATH)
        d1, d2, d3, d4 = st.columns(4)
        d1.metric("Scored predictions", f"{status.n_scored:,}")
        d2.metric("MAE — last 24h",
                  f"{status.mae_24h:.1f}" if not pd.isna(status.mae_24h) else "n/a",
                  help="Live MAE on predictions whose 30-min target has now arrived.")
        d3.metric("MAE — last 7d",
                  f"{status.mae_7d:.1f}" if not pd.isna(status.mae_7d) else "n/a")
        if status.drift_ratio is not None:
            d4.metric("Drift ratio (24h vs trained)",
                      f"{status.drift_ratio:.2f}×",
                      delta=("retrain recommended" if status.recommend_retrain else "stable"),
                      delta_color="inverse")
        else:
            d4.metric("Drift ratio", "n/a")
        if status.recommend_retrain:
            st.warning(status.recommendation)
        else:
            st.caption(status.recommendation)

        rolled = drift.rolling_mae_series(PREDICTIONS_LOG, UNIFIED_PARQUET, window="6h")
        if not rolled.empty:
            rolled = _decimate(rolled)
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(
                x=rolled["target_timestamp"], y=rolled["rolling_mae"],
                name="Rolling 6h MAE", line=dict(color="#9467bd", width=1.5),
            ))
            if status.trained_mae:
                fig2.add_hline(y=status.trained_mae, line=dict(color="green", dash="dash"),
                               annotation_text="trained MAE", annotation_position="top right")
                fig2.add_hline(y=status.trained_mae * 1.5, line=dict(color="orange", dash="dash"),
                               annotation_text="1.5× threshold", annotation_position="bottom right")
            fig2.update_layout(height=300, yaxis_title="MAE (mg/dL)", xaxis_title="UTC",
                               legend=dict(orientation="h"))
            st.plotly_chart(fig2, width="stretch")

        with st.expander("Recent prediction log (last 50 rows)"):
            st.dataframe(log.tail(50).iloc[::-1], width="stretch", hide_index=True)

    _live_panel()


# ---------- Log carbs section ------------------------------------------------

elif section == "Log carbs":
    st.title("Log carbs")
    st.caption(
        "Log meals, snacks, and low treatments. Bolused meals are pulled "
        "automatically from the pump — this form is for **carbs that the pump "
        "didn't see** (juice for a low, unbolused snacks). Stored append-only "
        "to `treatments.parquet`."
    )

    # --- Quick low-treatment buttons ---------------------------------------
    st.subheader("Quick low treatment")
    st.caption("One tap for the most common kid-sized low treatments. Logs immediately "
               "with the current time.")
    cols = st.columns(len(treatments_mod.LOW_TREATMENT_PRESETS))
    for col, preset in zip(cols, treatments_mod.LOW_TREATMENT_PRESETS):
        if col.button(preset["label"], width="stretch", key=f"low_{preset['carbs_g']}"):
            treatments_mod.append(
                TREATMENTS_PARQUET,
                kind="low_treatment",
                carbs_g=preset["carbs_g"],
                notes=preset["notes"],
            )
            st.success(f"Logged {preset['carbs_g']} g low treatment.")
            st.rerun()

    # --- Free-form log -----------------------------------------------------
    st.subheader("Log a meal or snack")
    with st.form("log_treatment", clear_on_submit=True):
        c1, c2, c3 = st.columns([1, 1, 2])
        kind = c1.selectbox("Kind", ["meal", "snack", "correction", "low_treatment", "other"], index=0)
        carbs_g = c2.number_input("Carbs (g)", min_value=0.0, max_value=300.0, value=15.0, step=1.0)
        notes = c3.text_input("Notes (optional)")
        c4, c5 = st.columns(2)
        use_now = c4.checkbox("Now", value=True, help="Uncheck to set a past time.")
        ts_input = c5.text_input(
            "Time (UTC, ISO 8601)",
            value="",
            disabled=use_now,
            help="e.g. 2026-04-25T14:30:00",
        )
        submitted = st.form_submit_button("Log", type="primary")
        if submitted:
            if carbs_g <= 0:
                st.error("Carbs must be greater than zero.")
            else:
                ts = None if use_now else ts_input
                try:
                    treatments_mod.append(TREATMENTS_PARQUET, timestamp=ts, kind=kind,
                                          carbs_g=carbs_g, notes=notes)
                    st.success(f"Logged {carbs_g:g} g ({kind}).")
                except Exception as e:  # noqa: BLE001
                    st.error(f"Log failed: {e}")

    # --- Recent log --------------------------------------------------------
    st.subheader("Recent treatments")
    recent = treatments_mod.load(TREATMENTS_PARQUET).tail(50).iloc[::-1]
    if recent.empty:
        st.info("No manual treatments logged yet.")
    else:
        st.dataframe(recent, width="stretch", hide_index=True)

        # Coverage stats vs pump-derived carbs
        if TANDEM_PARQUET.exists():
            tandem = _load(TANDEM_PARQUET)
            derived = treatments_mod.derive_from_tandem(tandem)
            unioned = treatments_mod.union(derived, treatments_mod.load(TREATMENTS_PARQUET))
            counts = unioned["source"].value_counts()
            st.caption("Sources currently feeding the carbs feature: "
                       + " · ".join(f"**{src}**: {n:,}" for src, n in counts.items()))


# ---------- Pipeline section -------------------------------------------------

elif section == "Pipeline":
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

    # Decimate the line series before handing to plotly. Boluses /
    # suspends are sparse markers — leave those un-decimated so we
    # don't drop events.
    view_line = _decimate(view, 2000)

    # --- Glucose + bolus chart -------------------------------------------
    fig = go.Figure()
    fig.add_hrect(y0=70, y1=180, fillcolor="green", opacity=0.08, line_width=0)
    fig.add_trace(go.Scatter(
        x=view_line["timestamp"], y=view_line["glucose_mg_dl"],
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
        x=view_line["timestamp"], y=view_line["glucose_mg_dl"],
        mode="lines", name="Glucose", yaxis="y", line=dict(color="#1f77b4", width=1.0),
    ))
    fig2.add_trace(go.Scatter(
        x=view_line["timestamp"], y=view_line["iob_units"],
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

elif section == "Stats":
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


# ---------- Model section ----------------------------------------------------

elif section == "Model":
    st.title("Model")
    st.caption(
        "Predicts glucose **30 minutes ahead** using XGBoost — gradient-boosted "
        "decision trees. See `docs/ML_PRIMER.md` for plain-English context."
    )

    if not UNIFIED_PARQUET.exists():
        st.info("No unified timeline yet. Run Phase 3 on the Pipeline tab first.")
        st.stop()

    # --- Build features --------------------------------------------------
    st.subheader("Features")
    with st.container(border=True):
        st.markdown(
            "Builds lag (5/10/15/30/60/120/240 min), rolling stats, IOB-context, "
            "and circadian features. Every row gets a `target_30m` label = glucose "
            "30 min later. **No future leakage**: features are functions of past "
            "data only."
        )
        col_a, col_b = st.columns([1, 3])
        with col_a:
            run_feats = st.button("Build features", type="primary")
        with col_b:
            st.write(f"Output → `{FEATURES_PARQUET.relative_to(REPO)}`")
        if run_feats:
            with st.spinner("Engineering features…"):
                try:
                    feats = feat_mod.run(UNIFIED_PARQUET, FEATURES_PARQUET)
                    st.success(f"Built {len(feats):,} rows × {len(feats.columns)} columns.")
                except Exception as e:  # noqa: BLE001
                    st.error(f"Feature build failed: {e}")

    # --- Train -----------------------------------------------------------
    st.subheader("Train")
    with st.container(border=True):
        st.markdown(
            "**Walk-forward split** in chronological order: 70% train / 15% validation "
            "/ 15% test. The model never sees the test window during training, which "
            "is the only honest way to estimate future-period accuracy. Early stopping "
            "is on the validation set."
        )
        col_a, col_b = st.columns([1, 3])
        with col_a:
            run_train = st.button(
                "Train XGBoost",
                type="primary",
                disabled=not FEATURES_PARQUET.exists(),
                help=None if FEATURES_PARQUET.exists() else "Build features first",
            )
        with col_b:
            st.write(f"Output → `{MODEL_DIR.relative_to(REPO)}/`")

        if run_train:
            with st.spinner("Training… (early stopping, usually 5-30s)"):
                try:
                    result = train_mod.run(FEATURES_PARQUET, MODEL_DIR)
                    m = result["metrics"]
                    st.success(
                        f"Done — test MAE = {m['mae']:.2f} mg/dL on "
                        f"{m['test_rows']:,} held-out rows."
                    )
                except Exception as e:  # noqa: BLE001
                    st.error(f"Training failed: {e}")

    # --- Metrics + Importance --------------------------------------------
    if METRICS_PATH.exists():
        st.subheader("Test-set performance")
        st.caption("All metrics computed on data the model **never saw** during training.")
        m = json.loads(METRICS_PATH.read_text())
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("MAE (mg/dL)", f"{m['mae']:.1f}",
                  help="Mean absolute error. Average miss in mg/dL. 12-18 is competitive for 30-min horizon.")
        c2.metric("RMSE (mg/dL)", f"{m['rmse']:.1f}",
                  help="Root mean squared error. Penalizes big misses more.")
        c3.metric("Within ±20 mg/dL", f"{m['pct_within_20']:.1f}%",
                  help="Fraction of predictions within 20 mg/dL of truth.")
        hr = m.get("hypo_recall")
        c4.metric("Hypo recall", f"{hr:.2f}" if isinstance(hr, (int, float)) and not pd.isna(hr) else "n/a",
                  help="Of actual lows (<70), fraction we predicted to be <80. Higher = safer.")
        with st.expander("Full metrics JSON"):
            st.json(m)

    # --- Run explainability ---------------------------------------------
    st.subheader("Explain")
    with st.container(border=True):
        st.markdown(
            "**SHAP values** decompose every prediction into per-feature contributions. "
            "Below: each feature's *average* impact on the prediction (mean |SHAP|). "
            "If a recent-glucose feature isn't on top, something is wrong."
        )
        col_a, col_b = st.columns([1, 3])
        with col_a:
            run_expl = st.button(
                "Compute SHAP",
                type="primary",
                disabled=not MODEL_PATH.exists(),
                help=None if MODEL_PATH.exists() else "Train the model first",
            )
        with col_b:
            st.write(f"Sample size: 2,000 rows (capped for speed).")
        if run_expl:
            with st.spinner("Computing SHAP…"):
                try:
                    explain_mod.run(FEATURES_PARQUET, MODEL_DIR, MODEL_DIR, sample_size=2000)
                    st.success("SHAP computed.")
                except Exception as e:  # noqa: BLE001
                    st.error(f"SHAP failed: {e}")

    if IMPORTANCE_PATH.exists():
        importance_full = _load(IMPORTANCE_PATH)
        st.bar_chart(importance_full.head(20).set_index("feature")["mean_abs_shap"], height=420)
        with st.expander("Full importance table"):
            st.dataframe(importance_full, width="stretch", hide_index=True)


# ---------- Predict section --------------------------------------------------

elif section == "Predict":
    st.title("Predict")
    st.caption(
        "Live 30-minute forecast for the most recent reading, plus a backtest of the "
        "model's predictions vs. what actually happened."
    )

    if not MODEL_PATH.exists() or not FEATURES_PARQUET.exists():
        st.info("Build features and train the model on the **Model** tab first.")
        st.stop()

    feats = _load(FEATURES_PARQUET)

    # --- Live forecast --------------------------------------------------
    st.subheader("Latest forecast")
    try:
        model, feature_cols = _load_model()
        latest = predict_mod.predict_latest(model, feature_cols, feats)
    except Exception as e:  # noqa: BLE001
        st.error(f"Prediction failed: {e}")
        st.stop()

    c1, c2, c3 = st.columns(3)
    c1.metric("Current glucose", f"{latest['current_glucose']:.0f} mg/dL")
    c2.metric("Predicted in 30 min", f"{latest['prediction_30m']:.0f} mg/dL",
              delta=f"{latest['predicted_change']:+.0f} mg/dL")
    c3.metric("As of (UTC)", latest["as_of"].split("+")[0].replace("T", " "))

    # --- Local explanation for that prediction --------------------------
    st.subheader("Why this prediction?")
    st.caption("Top features contributing to the latest forecast (positive = pushing up, negative = pulling down).")
    if LOCAL_EXPL_PATH.exists():
        local = json.loads(LOCAL_EXPL_PATH.read_text())
        contrib_df = pd.DataFrame(local["contributions"])
        if not contrib_df.empty:
            contrib_df["sign"] = contrib_df["shap"].apply(lambda v: "+" if v >= 0 else "−")
            st.dataframe(
                contrib_df[["feature", "value", "shap"]].rename(columns={"shap": "contribution_mg_dl"}),
                width="stretch", hide_index=True,
            )
            st.caption(f"Baseline (model average): **{local['baseline']:.1f} mg/dL** · "
                       f"Sum of contributions = prediction.")
    else:
        st.info("Run **Explain → Compute SHAP** on the Model tab to see the breakdown.")

    # --- Backtest chart -------------------------------------------------
    st.subheader("Backtest — predicted vs actual")
    st.markdown(
        "Predictions are computed for *every* row, including training data. The "
        "test window is the rightmost ~15% of the timeline; that's the slice that "
        "honestly measures generalization."
    )
    backtest = _cached_backtest(
        str(FEATURES_PARQUET), str(MODEL_DIR),
        FEATURES_PARQUET.stat().st_mtime, MODEL_PATH.stat().st_mtime,
    )
    backtest = backtest.dropna(subset=["actual_30m"])
    backtest_view = _decimate(backtest, 2000)
    if len(backtest) > len(backtest_view):
        st.caption(f"Decimated to {len(backtest_view):,} of {len(backtest):,} rows for the chart.")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=backtest_view["timestamp"], y=backtest_view["actual_30m"],
        name="Actual (t+30m)", line=dict(color="#1f77b4", width=1.2),
    ))
    fig.add_trace(go.Scatter(
        x=backtest_view["timestamp"], y=backtest_view["prediction_30m"],
        name="Predicted (t+30m)", line=dict(color="#d62728", width=1.0, dash="dot"),
    ))
    fig.add_hrect(y0=70, y1=180, fillcolor="green", opacity=0.06, line_width=0)
    fig.update_layout(
        height=420, yaxis_title="mg/dL", xaxis_title="UTC",
        legend=dict(orientation="h"),
    )
    st.plotly_chart(fig, width="stretch")

    overall_mae = backtest["abs_error"].mean()
    st.caption(f"Across all {len(backtest):,} rows: MAE = **{overall_mae:.2f} mg/dL** "
               "(includes training data — your test-set MAE on the Model tab is the honest one).")


# ---------- What-if section --------------------------------------------------

elif section == "What-if":
    st.title("What-if (counterfactuals)")
    st.markdown(
        "Ask the model: **what would it predict if a different action had just been taken?** "
        "We don't simulate the body — we mechanically modify the inputs (IOB, recent boluses, "
        "basal rate) and re-run the forecaster. The result is the model's *belief* about an "
        "action's effect, only as good as the patterns it has learned. "
        "See `docs/ML_PRIMER.md` for the limits."
    )

    if not MODEL_PATH.exists() or not FEATURES_PARQUET.exists():
        st.info("Need a trained model and a features table. Visit the **Model** tab first.")
        st.stop()

    feats = _load(FEATURES_PARQUET)
    model, feature_cols = _load_model()
    available = feats.dropna(subset=feature_cols)
    if available.empty:
        st.warning("No row has a complete feature vector yet — ingest more data.")
        st.stop()

    last_row = available.iloc[[-1]]
    current_glucose = float(last_row["glucose_mg_dl"].iloc[0])
    current_iob = float(last_row.get("iob_units", pd.Series([0])).iloc[0])
    as_of = str(last_row["timestamp"].iloc[0])

    c1, c2, c3 = st.columns(3)
    c1.metric("Current glucose", f"{current_glucose:.0f} mg/dL")
    c2.metric("Current IOB", f"{current_iob:.2f} U")
    c3.metric("As of (UTC)", as_of.split("+")[0].replace("T", " "))

    st.subheader("Configure scenarios")
    with st.container(border=True):
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.markdown("**Bolus doses to test (units)**")
            bolus_doses = st.multiselect(
                "Bolus units",
                options=[0.5, 1.0, 1.5, 2.0, 3.0, 4.0],
                default=[0.5, 1.0, 2.0, 3.0],
                label_visibility="collapsed",
            )
        with col_b:
            st.markdown("**Carbs to test (grams)**")
            carb_amounts = st.multiselect(
                "Carbs g",
                options=[4, 8, 15, 20, 30, 45, 60],
                default=[8, 15, 30],
                label_visibility="collapsed",
                help="Useful for low-treatment 'what if I give 8g now' planning.",
            )
        with col_c:
            st.markdown("**Suspend durations to test (minutes)**")
            suspend_durations = st.multiselect(
                "Suspend min",
                options=[15, 30, 45, 60, 90],
                default=[30, 60],
                label_visibility="collapsed",
            )

        if not any(c.startswith("carbs_sum_") for c in feature_cols):
            st.warning(
                "Carb scenarios will return 0 effect — the trained model has no "
                "carb features. Retrain after the next refresh that includes a "
                "carbs-bearing timeline to make these scenarios meaningful."
            )

    actions = (
        [counterfactual.Action("bolus", units=u) for u in bolus_doses]
        + [counterfactual.Action("carbs", grams=g) for g in carb_amounts]
        + [counterfactual.Action("suspend", minutes=m) for m in suspend_durations]
    )

    if not actions:
        st.info("Pick at least one action to compare.")
        st.stop()

    cf = counterfactual.simulate(model, feature_cols, last_row, actions)

    st.subheader("Predicted glucose at +30 min — by scenario")
    fig = go.Figure()
    colors = (
        ["#1f77b4"]
        + ["#ff7f0e"] * len(bolus_doses)
        + ["#9467bd"] * len(carb_amounts)
        + ["#2ca02c"] * len(suspend_durations)
    )
    fig.add_trace(go.Bar(
        x=cf["scenario"], y=cf["prediction_30m"],
        marker_color=colors[:len(cf)],
        text=[f"{v:.0f}" for v in cf["prediction_30m"]],
        textposition="outside",
    ))
    fig.add_hline(y=current_glucose, line=dict(color="gray", dash="dash"),
                  annotation_text=f"current ({current_glucose:.0f})",
                  annotation_position="top left")
    fig.add_hrect(y0=70, y1=180, fillcolor="green", opacity=0.06, line_width=0)
    fig.update_layout(height=420, yaxis_title="Predicted mg/dL at t+30m",
                      xaxis_title="", showlegend=False)
    st.plotly_chart(fig, width="stretch")

    st.subheader("Effect vs. doing nothing")
    cf_display = cf.copy()
    cf_display["delta_vs_baseline"] = cf_display["delta_vs_baseline"].round(1)
    cf_display["prediction_30m"] = cf_display["prediction_30m"].round(1)
    st.dataframe(cf_display, width="stretch", hide_index=True)
    st.caption(
        "Negative `delta_vs_baseline` means the action is predicted to bring glucose "
        "*lower* than no action would. Suspends only show small effects on the 30-min "
        "horizon because suspend's downstream impact is mostly > 30 min out."
    )
