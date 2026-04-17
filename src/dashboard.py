"""
Adaptive Drift Intelligence — exploration-first dashboard.

Built for two things:
  1. Clear visualization of drift severity across the feature space.
  2. Interactive exploration — filter, select, compare, drill down.

Loads `dashboard_artifacts.joblib` produced by `src/main.py`. Run with:
    streamlit run src/dashboard.py
"""

import os
import sys

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NAISC 2026 · Adaptive Drift Intelligence",
    layout="wide",
)

# ── Theme ────────────────────────────────────────────────────────────────────
BG = "#0d1117"
PANEL = "#161b22"
BORDER = "#30363d"
MUTED = "#8b949e"
ACCENT = "#58a6ff"
OK = "#3fb950"
WARN = "#d29922"
BAD = "#f85149"

SEVERITY_COLORS = {"high": BAD, "moderate": WARN, "low": OK, "unknown": MUTED}
SEVERITY_ORDER = {"high": 0, "moderate": 1, "low": 2, "unknown": 3}

st.markdown(
    f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&display=swap');
    .stApp {{ background-color: {BG}; color: #e6edf3; }}

    .kpi {{
        background: {PANEL};
        padding: 1.0rem 1.2rem;
        border-radius: 10px;
        border: 1px solid {BORDER};
        height: 100%;
    }}
    .kpi-label {{
        color: {MUTED};
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 1.4px;
        margin-bottom: 0.3rem;
    }}
    .kpi-val {{
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.7rem;
        font-weight: 700;
        color: {ACCENT};
    }}
    .kpi-sub {{
        color: {MUTED};
        font-size: 0.78rem;
        margin-top: 0.2rem;
    }}
    .pill {{
        display: inline-block;
        padding: 2px 10px;
        border-radius: 999px;
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1px;
    }}
    [data-testid="stSidebar"] {{
        background-color: {BG};
        border-right: 1px solid {BORDER};
    }}
    .block-container {{ padding-top: 3.5rem; }}
</style>
""",
    unsafe_allow_html=True,
)


def _plot_layout(fig, *, height=None):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e6edf3"),
        margin=dict(l=20, r=20, t=40, b=20),
    )
    if height:
        fig.update_layout(height=height)
    return fig


def _kpi(label: str, value: str, sub: str = "", color: str = ACCENT) -> str:
    return (
        f'<div class="kpi">'
        f'<div class="kpi-label">{label}</div>'
        f'<div class="kpi-val" style="color:{color}">{value}</div>'
        f'<div class="kpi-sub">{sub}</div>'
        f"</div>"
    )


# ── Artifact loading ─────────────────────────────────────────────────────────
ARTIFACT_PATH = "dashboard_artifacts.joblib"


@st.cache_resource
def load_artifacts(artifact_path: str):
    if not os.path.exists(artifact_path):
        return None
    art = joblib.load(artifact_path)
    art["train_pd"] = art["train"].to_pandas()
    art["test_pd"] = art["test"].to_pandas()
    art["drift_pd"] = art["drift_df"].to_pandas() if len(art["drift_df"]) else pd.DataFrame()
    return art


def run_pipeline(train_path: str, test_path: str) -> bool:
    """Run main.py as a subprocess. Returns True on success."""
    import subprocess
    # Resolve src/main.py relative to this file's location
    main_py = os.path.join(os.path.dirname(os.path.abspath(__file__)), "main.py")
    result = subprocess.run(
        [sys.executable, main_py,
         "--train_data_filepath", train_path,
         "--test_data_filepath", test_path],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    return result.returncode == 0, result.stdout, result.stderr


# ── Sidebar: file inputs ────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### NAISC 2026")
    st.caption("Adaptive Drift Intelligence")
    st.divider()

    st.markdown("**Data Files**")
    train_path = st.text_input("Train CSV", value="public_data/train.csv", key="train_path")
    test_path = st.text_input("Test CSV", value="public_data/test.csv", key="test_path")

    col_run, col_status = st.columns([1, 1])
    run_clicked = col_run.button("Run Pipeline", type="primary", use_container_width=True)

    has_artifacts = os.path.exists(ARTIFACT_PATH)
    if has_artifacts:
        import datetime
        mtime = os.path.getmtime(ARTIFACT_PATH)
        ts = datetime.datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M")
        col_status.caption(f"Artifacts: {ts}")
    else:
        col_status.caption("No artifacts yet")

    st.divider()

if run_clicked:
    if not os.path.exists(train_path):
        st.error(f"Train file not found: `{train_path}`")
        st.stop()
    if not os.path.exists(test_path):
        st.error(f"Test file not found: `{test_path}`")
        st.stop()

    with st.spinner("Running pipeline... this may take a minute."):
        ok, stdout, stderr = run_pipeline(train_path, test_path)

    if ok:
        st.success("Pipeline completed. Reloading dashboard...")
        load_artifacts.clear()
        st.rerun()
    else:
        st.error("Pipeline failed. See output below.")
        with st.expander("stdout", expanded=False):
            st.code(stdout)
        with st.expander("stderr", expanded=True):
            st.code(stderr)
        st.stop()

art = load_artifacts(ARTIFACT_PATH)

if art is None:
    st.title("Adaptive Drift Intelligence")
    st.info(
        "No pipeline artifacts found. Enter your train/test file paths in the sidebar and click **Run Pipeline**."
    )
    st.stop()


train: pd.DataFrame = art["train_pd"]
test: pd.DataFrame = art["test_pd"]
drift_df: pd.DataFrame = art["drift_pd"]
feature_types: dict = art["feature_types"]
norm_log: dict = art["norm_log"]
drop_cols: list = art["drop_cols"]
mitigation_log: list = art["mitigation_log"]
iter_log: dict = art["iter_log"]
new_features: list = art["new_feature_names"]
self_training: dict = art["self_training"]
baseline_metrics: dict = art["baseline_metrics"]
mitigated_metrics: dict = art["mitigated_metrics"]
runtimes: dict = art["runtimes"]

# Enrich drift_df with derived columns used everywhere.
if not drift_df.empty:
    drift_df = drift_df.copy()
    drift_df["drift_score"] = np.maximum(
        drift_df.get("psi", pd.Series(0, index=drift_df.index)).fillna(0),
        drift_df.get("jsd", pd.Series(0, index=drift_df.index)).fillna(0),
    )
    drift_df["_sev_rank"] = drift_df["severity"].map(SEVERITY_ORDER).fillna(4).astype(int)

# Attach feature importance from the mitigated model.
try:
    _model = joblib.load("model.joblib")
    _imp_df = pd.DataFrame(
        {"column": _model.feature_name_, "importance": _model.feature_importances_}
    )
    if not drift_df.empty:
        drift_df = drift_df.merge(_imp_df, on="column", how="left")
        drift_df["importance"] = drift_df["importance"].fillna(0)
except Exception:
    if not drift_df.empty:
        drift_df["importance"] = 0

# Session state for cross-tab feature selection.
if "focused_feature" not in st.session_state:
    st.session_state.focused_feature = None


severity_counts = drift_df["severity"].value_counts().to_dict() if not drift_df.empty else {}
n_total = len(drift_df)
n_high = severity_counts.get("high", 0)
n_moderate = severity_counts.get("moderate", 0)
n_low = severity_counts.get("low", 0)


# ── Sidebar: results summary (appended below file inputs) ────────────────────
with st.sidebar:
    st.markdown(
        f"**Dataset**\n- Train: `{art['train_shape'][0]:,} × {art['train_shape'][1]}`\n"
        f"- Test: `{art['test_shape'][0]:,} × {art['test_shape'][1]}`"
    )
    st.markdown(
        f"**Drift**\n- 🔴 High: **{n_high}**\n- 🟡 Moderate: **{n_moderate}**\n- 🟢 Low: **{n_low}**"
    )
    if baseline_metrics.get("test_auprc") is not None and mitigated_metrics.get("test_auprc") is not None:
        delta = (mitigated_metrics["test_auprc"] - baseline_metrics["test_auprc"]) / baseline_metrics["test_auprc"] * 100
        st.markdown(
            f"**AU-PRC (test)**\n- Baseline: `{baseline_metrics['test_auprc']:.4f}`\n"
            f"- Mitigated: `{mitigated_metrics['test_auprc']:.4f}`\n- Δ: **{delta:+.1f}%**"
        )
    st.divider()
    st.caption("Team: junyiding")


# ── Tabs ─────────────────────────────────────────────────────────────────────
tab_explore, tab_feat, tab_matrix, tab_results = st.tabs(
    ["Drift Explorer", "Feature Deep-Dive", "Metric Matrix", "Pipeline Results"]
)


# =============================================================================
# Tab 1 · Drift Explorer
# =============================================================================
def _render_explorer():
    st.title("Drift Explorer")
    st.caption(
        "Filter the feature space by severity, type, and drift magnitude. "
        "Click any row in the table to preview its distribution and metrics inline."
    )

    if drift_df.empty:
        st.info("No features analysed.")
        return

    # ── Filter bar ──
    max_drift = float(drift_df["drift_score"].max() or 1.0)
    types_available = sorted(drift_df["feature_type"].unique().tolist())

    fc1, fc2, fc3, fc4 = st.columns([1.5, 1.5, 1.8, 1.8])
    sev_sel = fc1.multiselect(
        "Severity",
        options=["high", "moderate", "low"],
        default=["high", "moderate", "low"],
    )
    type_sel = fc2.multiselect(
        "Feature type",
        options=types_available,
        default=types_available,
    )
    min_drift = fc3.slider(
        "Min drift score (PSI / JSD)",
        0.0,
        float(max(max_drift, 0.01)),
        0.0,
        step=0.01,
    )
    search = fc4.text_input("Search feature name", "", placeholder="e.g. Referr")

    mask = (
        drift_df["severity"].isin(sev_sel)
        & drift_df["feature_type"].isin(type_sel)
        & (drift_df["drift_score"] >= min_drift)
    )
    if search:
        mask &= drift_df["column"].str.contains(search, case=False, na=False)

    filtered = drift_df[mask].sort_values(["_sev_rank", "drift_score"], ascending=[True, False])

    # ── KPI summary of filtered view ──
    f_high = int((filtered["severity"] == "high").sum())
    f_mod = int((filtered["severity"] == "moderate").sum())
    f_low = int((filtered["severity"] == "low").sum())

    k1, k2, k3, k4 = st.columns(4)
    k1.markdown(
        _kpi("Shown", f"{len(filtered)}", f"of {n_total} features"),
        unsafe_allow_html=True,
    )
    k2.markdown(
        _kpi("High", str(f_high), "severe shift", color=BAD),
        unsafe_allow_html=True,
    )
    k3.markdown(
        _kpi("Moderate", str(f_mod), "notable shift", color=WARN),
        unsafe_allow_html=True,
    )
    k4.markdown(
        _kpi("Low", str(f_low), "stable", color=OK),
        unsafe_allow_html=True,
    )

    st.markdown("")

    if filtered.empty:
        st.warning("No features match the current filters.")
        return

    # ── Scatter ──
    st.subheader("Severity map")
    st.caption("Each point is a feature. Horizontal axis = model importance, vertical axis = drift score.")
    fig = px.scatter(
        filtered,
        x="importance",
        y="drift_score",
        color="severity",
        color_discrete_map=SEVERITY_COLORS,
        size=np.clip(filtered["drift_score"].values * 12 + 5, 5, 50),
        hover_name="column",
        hover_data={
            "feature_type": True,
            "severity": True,
            "drift_score": ":.4f",
            "importance": ":.0f",
        },
        category_orders={"severity": ["high", "moderate", "low"]},
        labels={"importance": "Model importance (gain)", "drift_score": "Drift score"},
    )
    fig.update_traces(marker=dict(line=dict(width=1, color="#0d1117")))
    st.plotly_chart(_plot_layout(fig, height=380), use_container_width=True)

    # ── Selectable table ──
    st.subheader("Feature list")
    st.caption("Click a row to preview. Sort by any column.")

    display_cols = ["column", "feature_type", "severity", "drift_score", "importance"]
    for extra in ("ks_pval_bh", "chi2_pval_bh", "train_null_pct", "test_null_pct"):
        if extra in filtered.columns:
            display_cols.append(extra)

    table_view = filtered[display_cols].reset_index(drop=True)

    try:
        event = st.dataframe(
            table_view,
            use_container_width=True,
            hide_index=True,
            height=340,
            on_select="rerun",
            selection_mode="single-row",
            key="explorer_table",
        )
        selected_rows = list(getattr(event.selection, "rows", []) or [])
    except TypeError:
        # Fallback for older Streamlit versions without dataframe selection.
        st.dataframe(table_view, use_container_width=True, hide_index=True, height=340)
        selected_rows = []

    focused_col = None
    if selected_rows:
        focused_col = str(table_view.iloc[selected_rows[0]]["column"])
        st.session_state.focused_feature = focused_col
    elif st.session_state.focused_feature in filtered["column"].values:
        focused_col = st.session_state.focused_feature

    st.markdown("---")
    st.subheader("Focused feature preview")
    if focused_col is None:
        st.info("Select a row in the table above to preview a feature here.")
        return

    _render_feature_preview(focused_col, compact=True)


def _render_feature_preview(feat: str, *, compact: bool = False) -> None:
    """Render distribution + key metrics for a single feature.

    In compact mode, shows a single distribution chart and a 3-metric header.
    In full mode, shows stats tables and the full metric row.
    """
    if feat not in drift_df["column"].values:
        st.warning(f"Feature `{feat}` not in drift table.")
        return

    row = drift_df[drift_df["column"] == feat].iloc[0]
    ftype = row["feature_type"]

    # Metric highlights
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(
        _kpi("Feature", feat, ftype.capitalize()),
        unsafe_allow_html=True,
    )
    c2.markdown(
        _kpi(
            "Severity",
            row["severity"].capitalize(),
            "post BH-FDR",
            color=SEVERITY_COLORS.get(row["severity"], MUTED),
        ),
        unsafe_allow_html=True,
    )
    if ftype == "numerical":
        psi = row.get("psi")
        c3.markdown(
            _kpi("PSI", f"{psi:.4f}" if pd.notna(psi) else "—", "0.1 ≈ moderate · 0.25 high"),
            unsafe_allow_html=True,
        )
        kd = row.get("ks_stat")
        kp = row.get("ks_pval_bh", row.get("ks_pval"))
        c4.markdown(
            _kpi(
                "KS",
                f"D={kd:.3f}" if pd.notna(kd) else "—",
                f"p={kp:.2e}" if pd.notna(kp) else "",
            ),
            unsafe_allow_html=True,
        )
    else:
        jsd = row.get("jsd")
        c3.markdown(
            _kpi("JSD", f"{jsd:.4f}" if pd.notna(jsd) else "—", "0 = identical"),
            unsafe_allow_html=True,
        )
        cv = row.get("cramers_v")
        cp = row.get("chi2_pval_bh", row.get("chi2_pval"))
        c4.markdown(
            _kpi(
                "Cramér's V",
                f"{cv:.3f}" if pd.notna(cv) else "—",
                f"χ² p={cp:.2e}" if pd.notna(cp) else "",
            ),
            unsafe_allow_html=True,
        )

    # Distribution plot
    if feat in train.columns and feat in test.columns:
        if ftype == "numerical":
            tr = pd.to_numeric(train[feat], errors="coerce").dropna()
            te = pd.to_numeric(test[feat], errors="coerce").dropna()
            combined = pd.concat(
                [
                    pd.DataFrame({"value": tr, "dataset": "Train"}),
                    pd.DataFrame({"value": te, "dataset": "Test"}),
                ]
            )
            fig = px.ecdf(
                combined,
                x="value",
                color="dataset",
                color_discrete_map={"Train": ACCENT, "Test": BAD},
                title=f"Cumulative distribution — {feat}",
            )
            st.plotly_chart(_plot_layout(fig, height=320), use_container_width=True)
            if not compact:
                sc1, sc2 = st.columns(2)
                with sc1:
                    st.markdown("**Train statistics**")
                    st.dataframe(tr.describe().round(4).to_frame("train"), use_container_width=True)
                with sc2:
                    st.markdown("**Test statistics**")
                    st.dataframe(te.describe().round(4).to_frame("test"), use_container_width=True)
        else:
            cats = sorted(
                set(train[feat].dropna().astype(str).unique())
                | set(test[feat].dropna().astype(str).unique())
            )
            tr_p = train[feat].astype(str).value_counts(normalize=True).reindex(cats, fill_value=0)
            te_p = test[feat].astype(str).value_counts(normalize=True).reindex(cats, fill_value=0)
            cat_df = pd.DataFrame(
                {
                    "category": cats * 2,
                    "proportion": tr_p.tolist() + te_p.tolist(),
                    "dataset": ["Train"] * len(cats) + ["Test"] * len(cats),
                }
            )
            fig = px.bar(
                cat_df,
                x="category",
                y="proportion",
                color="dataset",
                barmode="group",
                color_discrete_map={"Train": ACCENT, "Test": BAD},
                title=f"Category proportions — {feat}",
            )
            st.plotly_chart(_plot_layout(fig, height=320), use_container_width=True)
    else:
        st.info("Feature not available in both train and test columns.")

    # Mitigation trace
    actions = []
    if feat in norm_log:
        actions.append(("Normalization", "; ".join(norm_log[feat])))
    mit_entry = next((m for m in mitigation_log if m.get("column") == feat), None)
    if mit_entry:
        actions.append(
            (
                "Progressive mitigation",
                f"{mit_entry.get('strategy_applied')} "
                f"({mit_entry.get('original_severity')} → {mit_entry.get('final_severity')})",
            )
        )
    if feat in iter_log:
        actions.append(("Iterative pass", iter_log[feat]))
    if feat in drop_cols:
        actions.append(("Final state", "Dropped"))

    if not compact and actions:
        st.markdown("**Mitigation trace**")
        st.dataframe(
            pd.DataFrame(actions, columns=["stage", "action"]),
            use_container_width=True,
            hide_index=True,
        )
    elif compact and actions:
        st.caption("**Pipeline actions:** " + " · ".join(f"_{s}_: {a}" for s, a in actions))

    if not compact:
        st.markdown("---")
        st.subheader("Full metric row")
        mr = row.drop(["_sev_rank"], errors="ignore").to_frame("value")
        mr["value"] = mr["value"].astype(str)
        st.dataframe(mr, use_container_width=True)


# =============================================================================
# Tab 2 · Feature Deep-Dive  (with compare mode)
# =============================================================================
def _render_deep_dive():
    st.title("Feature Deep-Dive")
    st.caption("Inspect one feature in detail, or select a second feature to compare distributions side-by-side.")

    if drift_df.empty:
        st.info("No features available.")
        return

    ordered = drift_df.sort_values(["_sev_rank", "drift_score"], ascending=[True, False])
    options = ordered["column"].tolist()

    default_idx = 0
    if st.session_state.focused_feature in options:
        default_idx = options.index(st.session_state.focused_feature)

    c1, c2 = st.columns(2)
    primary = c1.selectbox("Primary feature", options, index=default_idx, key="dd_primary")
    compare_options = ["(none)"] + [o for o in options if o != primary]
    secondary = c2.selectbox("Compare with", compare_options, index=0, key="dd_secondary")

    st.markdown("---")

    if secondary == "(none)":
        _render_feature_preview(primary, compact=False)
    else:
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown(f"### {primary}")
            _render_feature_preview(primary, compact=True)
        with col_b:
            st.markdown(f"### {secondary}")
            _render_feature_preview(secondary, compact=True)


# =============================================================================
# Tab 3 · Metric Matrix
# =============================================================================
def _render_matrix():
    st.title("Metric Matrix")
    st.caption(
        "See how each feature scores across every drift metric at once. "
        "Values are min-max normalized per column so colors are comparable across scales."
    )

    if drift_df.empty:
        st.info("No features to plot.")
        return

    # Metrics to include (presence-guarded); mix numerical and categorical.
    metric_cols = [
        ("psi", "PSI"),
        ("ks_stat", "KS D"),
        ("wasserstein_norm", "W·norm"),
        ("iqr_ratio", "IQR ratio"),
        ("tail_divergence", "Tail div"),
        ("jsd", "JSD"),
        ("cramers_v", "Cramér's V"),
    ]
    metric_cols = [(c, label) for c, label in metric_cols if c in drift_df.columns]

    df = drift_df[["column", "feature_type", "severity", "_sev_rank"] + [c for c, _ in metric_cols]].copy()

    # Null-rate delta for completeness
    if "train_null_pct" in drift_df.columns and "test_null_pct" in drift_df.columns:
        df["null_delta"] = (
            drift_df["test_null_pct"].fillna(0) - drift_df["train_null_pct"].fillna(0)
        ).abs()
        metric_cols.append(("null_delta", "|Δ null%|"))

    # Sort by severity then drift_score
    df = df.merge(drift_df[["column", "drift_score"]], on="column").sort_values(
        ["_sev_rank", "drift_score"], ascending=[True, False]
    )

    view_mode = st.radio(
        "View",
        ["Heatmap", "Parallel coordinates"],
        horizontal=True,
        label_visibility="collapsed",
    )

    if view_mode == "Heatmap":
        # Min-max normalize each metric column independently
        matrix = df.set_index("column")[[c for c, _ in metric_cols]].astype(float)
        norm = matrix.copy()
        for c in norm.columns:
            col = norm[c]
            if col.notna().sum() == 0:
                continue
            mn, mx = col.min(skipna=True), col.max(skipna=True)
            if mx > mn:
                norm[c] = (col - mn) / (mx - mn)
            else:
                norm[c] = 0.0
        norm.columns = [label for _, label in metric_cols]

        # Hover text with raw values
        raw = matrix.copy()
        raw.columns = [label for _, label in metric_cols]
        hover = np.empty_like(norm.values, dtype=object)
        for i, row_col in enumerate(norm.index):
            for j, metric_label in enumerate(norm.columns):
                raw_val = raw.iat[i, j]
                hover[i, j] = (
                    f"<b>{row_col}</b><br>{metric_label}: "
                    f"{raw_val:.4f}" if pd.notna(raw_val) else f"<b>{row_col}</b><br>{metric_label}: n/a"
                )

        fig = go.Figure(
            go.Heatmap(
                z=norm.values,
                x=norm.columns,
                y=norm.index,
                colorscale="Inferno",
                zmin=0,
                zmax=1,
                hoverinfo="text",
                text=hover,
                colorbar=dict(title="normalized"),
            )
        )
        height = max(480, 22 * len(norm.index))
        fig.update_layout(yaxis=dict(autorange="reversed"))
        st.plotly_chart(_plot_layout(fig, height=height), use_container_width=True)
        st.caption(
            "Brighter = higher on that metric relative to other features. Hover for raw values."
        )

    else:  # Parallel coordinates
        pc_df = df[["column", "severity", "_sev_rank"] + [c for c, _ in metric_cols]].copy()
        pc_df = pc_df.rename(columns={c: label for c, label in metric_cols})
        pc_df = pc_df.dropna(subset=[label for _, label in metric_cols], how="all")

        sev_to_num = {"high": 2, "moderate": 1, "low": 0, "unknown": -1}
        pc_df["_sev_num"] = pc_df["severity"].map(sev_to_num).fillna(-1)

        dims = [
            dict(label=label, values=pc_df[label].fillna(0))
            for _, label in metric_cols
        ]

        fig = go.Figure(
            go.Parcoords(
                line=dict(
                    color=pc_df["_sev_num"],
                    colorscale=[[0, OK], [0.5, WARN], [1, BAD]],
                    cmin=0,
                    cmax=2,
                    showscale=True,
                    colorbar=dict(
                        title="severity",
                        tickvals=[0, 1, 2],
                        ticktext=["low", "moderate", "high"],
                    ),
                ),
                dimensions=dims,
            )
        )
        st.plotly_chart(_plot_layout(fig, height=500), use_container_width=True)
        st.caption(
            "Each line is a feature traced across all metrics, colored by severity. "
            "Features with one-metric spikes (e.g. high PSI but low KS) stand out as non-parallel lines."
        )


# =============================================================================
# Tab 4 · Pipeline Results
# =============================================================================
def _render_results():
    st.title("Pipeline Results")
    st.caption(
        "Condensed view of what the pipeline did and how the model performed. "
        "Full mitigation / self-training / runtime details are captured below."
    )

    # Metrics
    c1, c2, c3, c4 = st.columns(4)
    base_test = baseline_metrics.get("test_auprc")
    mit_test = mitigated_metrics.get("test_auprc")
    if base_test is not None and mit_test is not None:
        delta_pct = (mit_test - base_test) / base_test * 100
    else:
        delta_pct = None
    c1.markdown(
        _kpi(
            "AU-PRC (test, mitigated)",
            f"{mit_test:.4f}" if mit_test is not None else "—",
            f"{delta_pct:+.1f}% vs baseline" if delta_pct is not None else "test labels N/A",
            color=OK if (delta_pct or 0) > 0 else MUTED,
        ),
        unsafe_allow_html=True,
    )
    c2.markdown(
        _kpi(
            "AU-PRC (test, baseline)",
            f"{base_test:.4f}" if base_test is not None else "—",
            "no drift handling",
            color=MUTED,
        ),
        unsafe_allow_html=True,
    )
    c3.markdown(
        _kpi(
            "Pseudo-labels",
            f"{self_training.get('n_pseudo', 0):,}",
            f"{self_training.get('n_pos', 0):,}+ · {self_training.get('n_neg', 0):,}−",
        ),
        unsafe_allow_html=True,
    )
    c4.markdown(
        _kpi(
            "Runtime",
            f"{runtimes['total']:.2f}s",
            "within 300s budget",
            color=ACCENT,
        ),
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # AU-PRC bars + runtime side by side
    pc1, pc2 = st.columns([1, 1])

    rows = []
    for tag, m in [("Baseline", baseline_metrics), ("Mitigated", mitigated_metrics)]:
        rows.append({"model": tag, "split": "train", "au_prc": m.get("train_auprc")})
        rows.append({"model": tag, "split": "test", "au_prc": m.get("test_auprc")})
    mdf = pd.DataFrame(rows).dropna()

    with pc1:
        st.subheader("AU-PRC")
        if not mdf.empty:
            fig = px.bar(
                mdf,
                x="split",
                y="au_prc",
                color="model",
                barmode="group",
                text=mdf["au_prc"].round(4),
                color_discrete_map={"Baseline": MUTED, "Mitigated": ACCENT},
            )
            fig.update_traces(textposition="outside")
            fig.update_layout(yaxis=dict(range=[0, 1]), xaxis_title="", yaxis_title="AU-PRC")
            st.plotly_chart(_plot_layout(fig, height=320), use_container_width=True)

    with pc2:
        st.subheader("Runtime per stage")
        rt_df = pd.DataFrame(
            {
                "stage": ["Ingest", "Drift", "Mitigation", "Feat-Eng", "Model"],
                "seconds": [
                    runtimes["ingest"],
                    runtimes["drift"],
                    runtimes["mitigation"],
                    runtimes["feature_eng"],
                    runtimes["model"],
                ],
            }
        )
        fig = px.bar(
            rt_df,
            x="seconds",
            y="stage",
            orientation="h",
            color="seconds",
            color_continuous_scale="Blues",
            text=rt_df["seconds"].round(2),
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(yaxis=dict(autorange="reversed"), coloraxis_showscale=False)
        st.plotly_chart(_plot_layout(fig, height=320), use_container_width=True)

    st.markdown("---")
    st.subheader("Mitigation actions per feature")
    st.caption(
        "One row per feature that the pipeline touched. Combines data-quality normalization, "
        "progressive mitigation, importance-aware iterative pass, and final drop decisions."
    )

    touched = set(norm_log.keys()) | {m["column"] for m in mitigation_log} | set(iter_log.keys()) | set(drop_cols)
    rows = []
    for col in sorted(touched):
        norm_actions = "; ".join(norm_log.get(col, [])) or ""
        mit_entry = next((m for m in mitigation_log if m["column"] == col), None)
        prog = (
            f"{mit_entry['strategy_applied']} "
            f"({mit_entry['original_severity']}→{mit_entry['final_severity']})"
            if mit_entry
            else ""
        )
        iter_action = iter_log.get(col, "")
        dropped = "yes" if col in drop_cols else ""
        rows.append(
            {
                "feature": col,
                "normalization": norm_actions,
                "progressive": prog,
                "iterative": iter_action,
                "dropped": dropped,
            }
        )
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True, height=360)
    else:
        st.info("No features required mitigation.")

    st.markdown("---")
    cc1, cc2 = st.columns(2)
    with cc1:
        st.subheader("Self-training")
        st.markdown(
            f"- Pseudo-labeled samples: **{self_training.get('n_pseudo', 0):,}** "
            f"({self_training.get('n_pos', 0):,} positive, {self_training.get('n_neg', 0):,} negative)\n"
            f"- Confidence gates: `P ≥ {self_training.get('conf_high', 0.95)}` → 1, "
            f"`P ≤ {self_training.get('conf_low', 0.05)}` → 0\n"
            f"- Augmented train size: **{self_training.get('augmented_train_size', 0):,}**\n"
            f"- Single iteration to avoid confirmation bias."
        )
    with cc2:
        st.subheader("Feature engineering")
        st.markdown(f"- **{len(new_features)}** engineered features added")
        if new_features:
            with st.expander("Show all generated features"):
                st.write(", ".join(f"`{f}`" for f in new_features))


# ── Dispatch ─────────────────────────────────────────────────────────────────
with tab_explore:
    _render_explorer()
with tab_feat:
    _render_deep_dive()
with tab_matrix:
    _render_matrix()
with tab_results:
    _render_results()

st.markdown("---")
st.caption("NAISC Singtel 2026 · junyiding")
