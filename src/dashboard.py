import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from profiler import load_data, detect_feature_types
from drift_detector import run_drift_analysis
from mitigation import apply_all_mitigations, normalize_data_quality

# ── Page config & custom CSS ──────────────────────────────────────────────
st.set_page_config(
    page_title="NAISC Adaptive Drift Intelligence",
    layout="wide",
)

TRAIN_COLOR = "#636EFA"
TEST_COLOR = "#EF553B"
PLOTLY_TEMPLATE = "plotly_white"

st.markdown("""
<style>
div[data-testid="stMetric"] {
    background-color: #f8f9fa;
    border-radius: 8px;
    padding: 12px 16px;
    border-left: 4px solid #636EFA;
    box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    color: #1a1a1a;
}
div[data-testid="stMetric"] label,
div[data-testid="stMetric"] [data-testid="stMetricValue"],
div[data-testid="stMetric"] [data-testid="stMetricDelta"] {
    color: #1a1a1a;
}
.pipeline-card {
    background: linear-gradient(135deg, #f8f9fa, #ffffff);
    border: 1px solid #e0e0e0;
    border-radius: 12px;
    padding: 20px;
    text-align: center;
    box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    min-height: 140px;
    color: #1a1a1a;
}
.pipeline-card h4 { margin: 0 0 8px 0; color: #1a1a1a; }
.pipeline-card p { margin: 0; font-size: 0.9rem; color: #555; }
.badge-high { background: #ffcccc; color: #cc0000; padding: 2px 10px; border-radius: 12px; font-weight: 600; font-size: 0.85rem; }
.badge-moderate { background: #fff3cc; color: #cc8800; padding: 2px 10px; border-radius: 12px; font-weight: 600; font-size: 0.85rem; }
.badge-low { background: #ccffcc; color: #006600; padding: 2px 10px; border-radius: 12px; font-weight: 600; font-size: 0.85rem; }
button[data-baseweb="tab"] { font-size: 1.1rem; font-weight: 600; }
#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────
st.sidebar.markdown("### Team **junyiding**")
st.sidebar.markdown("NAISC Singtel 2026")
st.sidebar.divider()
st.sidebar.header("Data Sources")
default_train = "public_data/train.csv"
default_test = "public_data/test.csv"
train_path = st.sidebar.text_input("Train CSV path", value=default_train)
test_path = st.sidebar.text_input("Test CSV path", value=default_test)


# ── Data loading (cached) ────────────────────────────────────────────────
def _to_pd(obj):
    """Convert polars DataFrame/Series to pandas for dashboard use."""
    import polars as pl
    if isinstance(obj, pl.DataFrame):
        return obj.to_pandas()
    if isinstance(obj, pl.Series):
        return obj.to_pandas()
    return obj


@st.cache_data
def load_all_data(train_path: str, test_path: str):
    train_pl, test_pl = load_data(train_path, test_path)
    feature_types = detect_feature_types(train_pl)
    drift_df_pl = run_drift_analysis(train_pl, test_pl, feature_types)
    train_m_pl, test_m_pl, drop_cols, mitigation_log, norm_log = apply_all_mitigations(
        train_pl, test_pl, drift_df_pl, feature_types
    )

    # Convert polars → pandas for dashboard visualization
    train = _to_pd(train_pl)
    test = _to_pd(test_pl)
    drift_df = _to_pd(drift_df_pl)
    train_m = _to_pd(train_m_pl)
    test_m = _to_pd(test_m_pl)

    # Model metrics (if model files exist)
    baseline_metrics, mitigated_metrics = {}, {}
    pred_baseline, pred_mitigated = None, None

    try:
        import joblib
        from sklearn.metrics import average_precision_score
        from model import prepare_features

        target_col = feature_types["target"][0] if feature_types["target"] else "ChurnStatus"

        if os.path.exists("model_baseline.joblib"):
            model_b = joblib.load("model_baseline.joblib")
            X_train_b, y_train_b, X_test_b, _ = prepare_features(train, test, feature_types)
            train_proba_b = model_b.predict_proba(X_train_b)[:, 1]
            baseline_metrics["train_auprc"] = average_precision_score(y_train_b, train_proba_b)
            if target_col in test.columns and test[target_col].notna().any():
                test_proba_b = model_b.predict_proba(X_test_b)[:, 1]
                y_test = (test[target_col] == "Yes").astype(int)
                baseline_metrics["test_auprc"] = average_precision_score(y_test, test_proba_b)

        if os.path.exists("model_mitigated.joblib"):
            model_m_model = joblib.load("model_mitigated.joblib")
            X_train_m, y_train_m, X_test_m, _ = prepare_features(
                train_m, test_m, feature_types, extra_drop_cols=drop_cols
            )
            train_proba_m = model_m_model.predict_proba(X_train_m)[:, 1]
            mitigated_metrics["train_auprc"] = average_precision_score(y_train_m, train_proba_m)
            if target_col in test_m.columns and test_m[target_col].notna().any():
                test_proba_m = model_m_model.predict_proba(X_test_m)[:, 1]
                y_test_m = (test_m[target_col] == "Yes").astype(int)
                mitigated_metrics["test_auprc"] = average_precision_score(y_test_m, test_proba_m)

        if os.path.exists("prediction_baseline.csv"):
            pred_baseline = pd.read_csv("prediction_baseline.csv")
        if os.path.exists("prediction_mitigated.csv"):
            pred_mitigated = pd.read_csv("prediction_mitigated.csv")
    except Exception:
        pass

    return {
        "train": train, "test": test,
        "feature_types": feature_types, "drift_df": drift_df,
        "train_m": train_m, "test_m": test_m,
        "drop_cols": drop_cols,
        "mitigation_log": mitigation_log, "norm_log": norm_log,
        "baseline_metrics": baseline_metrics, "mitigated_metrics": mitigated_metrics,
        "pred_baseline": pred_baseline, "pred_mitigated": pred_mitigated,
    }


if not os.path.exists(train_path) or not os.path.exists(test_path):
    st.error("Please provide valid file paths in the sidebar.")
    st.stop()

data = load_all_data(train_path, test_path)
train = data["train"]
test = data["test"]
feature_types = data["feature_types"]
drift_df = data["drift_df"]
train_m = data["train_m"]
test_m = data["test_m"]
drop_cols = data["drop_cols"]
mitigation_log = data["mitigation_log"]
baseline_metrics = data["baseline_metrics"]
mitigated_metrics = data["mitigated_metrics"]
pred_baseline = data["pred_baseline"]
pred_mitigated = data["pred_mitigated"]

target_col = feature_types["target"][0] if feature_types["target"] else "ChurnStatus"

# ── Title ─────────────────────────────────────────────────────────────────
st.title("NAISC Adaptive Drift Intelligence")

# ══════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════
tab_data, tab_drift, tab_mitigation = st.tabs(
    ["Data Overview", "Drift Analysis", "Mitigation & Model Comparison"]
)

# ══════════════════════════════════════════════════════════════════════════
# TAB 1 : DATA OVERVIEW
# ══════════════════════════════════════════════════════════════════════════
with tab_data:
    st.header("Dataset Summary")

    n_features = sum(len(v) for k, v in feature_types.items() if k not in ("id", "target"))
    train_churn = train[target_col].value_counts(normalize=True).get("Yes", 0)
    test_churn = test[target_col].value_counts(normalize=True).get("Yes", 0) if target_col in test.columns else None

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Train Rows", f"{len(train):,}")
    c2.metric("Test Rows", f"{len(test):,}")
    c3.metric("Features", n_features)
    c4.metric("Train Churn Rate", f"{train_churn:.1%}")
    c5.metric("Test Churn Rate", f"{test_churn:.1%}" if test_churn is not None else "N/A")

    st.divider()

    # ── Feature types: table + donut ──
    st.subheader("Feature Types")
    col_table, col_donut = st.columns([1, 2])

    type_counts = {k: len(v) for k, v in feature_types.items() if k not in ("id", "target") and len(v) > 0}
    with col_table:
        type_df = pd.DataFrame(
            [{"Type": k.capitalize(), "Count": v} for k, v in type_counts.items()]
        )
        st.dataframe(type_df, use_container_width=True, hide_index=True)

    with col_donut:
        fig = go.Figure(go.Pie(
            labels=[k.capitalize() for k in type_counts],
            values=list(type_counts.values()),
            hole=0.45,
            marker_colors=["#636EFA", "#EF553B", "#00CC96", "#AB63FA"],
        ))
        fig.update_layout(
            template=PLOTLY_TEMPLATE, height=300,
            margin=dict(t=20, b=20, l=20, r=20),
        )
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ── Target distribution ──
    st.subheader("Target Distribution")
    col_left, col_right = st.columns(2)

    train_vc = train[target_col].value_counts().reset_index()
    train_vc.columns = [target_col, "count"]
    train_vc["dataset"] = "Train"

    if target_col in test.columns:
        test_vc = test[target_col].value_counts().reset_index()
        test_vc.columns = [target_col, "count"]
        test_vc["dataset"] = "Test"
        combined_vc = pd.concat([train_vc, test_vc], ignore_index=True)
    else:
        combined_vc = train_vc

    with col_left:
        fig = px.bar(
            combined_vc, x=target_col, y="count", color="dataset", barmode="group",
            color_discrete_map={"Train": TRAIN_COLOR, "Test": TEST_COLOR},
            title="Churn Counts",
        )
        fig.update_layout(template=PLOTLY_TEMPLATE, height=350)
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        # Proportional stacked bar
        train_prop = train[target_col].value_counts(normalize=True).reset_index()
        train_prop.columns = [target_col, "proportion"]
        train_prop["dataset"] = "Train"
        if target_col in test.columns:
            test_prop = test[target_col].value_counts(normalize=True).reset_index()
            test_prop.columns = [target_col, "proportion"]
            test_prop["dataset"] = "Test"
            combined_prop = pd.concat([train_prop, test_prop], ignore_index=True)
        else:
            combined_prop = train_prop

        fig = px.bar(
            combined_prop, x="dataset", y="proportion", color=target_col, barmode="stack",
            title="Churn Rate Proportions",
            color_discrete_map={"Yes": "#EF553B", "No": "#636EFA"},
        )
        fig.update_layout(template=PLOTLY_TEMPLATE, height=350)
        st.plotly_chart(fig, use_container_width=True)

    # ── Feature stats ──
    num_cols = feature_types.get("numerical", [])
    if num_cols:
        with st.expander("Numerical Feature Statistics"):
            st.dataframe(
                train[num_cols].describe().T.round(3),
                use_container_width=True,
            )

# ══════════════════════════════════════════════════════════════════════════
# TAB 2 : DRIFT ANALYSIS
# ══════════════════════════════════════════════════════════════════════════
with tab_drift:
    st.header("Drift Analysis")

    # ── Severity cards ──
    n_high = (drift_df["severity"] == "high").sum()
    n_moderate = (drift_df["severity"] == "moderate").sum()
    n_low = (drift_df["severity"] == "low").sum()

    sc1, sc2, sc3 = st.columns(3)
    sc1.markdown(
        f'<div style="background:#ffcccc; border-radius:8px; padding:16px; text-align:center;">'
        f'<h2 style="margin:0; color:#cc0000;">{n_high}</h2>'
        f'<p style="margin:0; color:#cc0000; font-weight:600;">High Severity</p></div>',
        unsafe_allow_html=True,
    )
    sc2.markdown(
        f'<div style="background:#fff3cc; border-radius:8px; padding:16px; text-align:center;">'
        f'<h2 style="margin:0; color:#cc8800;">{n_moderate}</h2>'
        f'<p style="margin:0; color:#cc8800; font-weight:600;">Moderate Severity</p></div>',
        unsafe_allow_html=True,
    )
    sc3.markdown(
        f'<div style="background:#ccffcc; border-radius:8px; padding:16px; text-align:center;">'
        f'<h2 style="margin:0; color:#006600;">{n_low}</h2>'
        f'<p style="margin:0; color:#006600; font-weight:600;">Low Severity</p></div>',
        unsafe_allow_html=True,
    )

    st.divider()

    # ── Drift overview bar chart ──
    st.subheader("Drift Overview")

    severity_order = {"high": 3, "moderate": 2, "low": 1}
    overview = drift_df.copy()
    overview["severity_rank"] = overview["severity"].map(severity_order)

    def get_drift_score(row):
        if row["feature_type"] == "numerical":
            return row.get("psi", 0) or 0
        return row.get("jsd", 0) or 0

    overview["drift_score"] = overview.apply(get_drift_score, axis=1)
    overview = overview.sort_values("drift_score", ascending=True)

    fig = px.bar(
        overview, y="column", x="drift_score", color="severity", orientation="h",
        color_discrete_map={"high": "#EF553B", "moderate": "#FECB52", "low": "#00CC96"},
        title="Drift Score by Feature (PSI for numerical, JSD for categorical/temporal)",
        labels={"column": "Feature", "drift_score": "Drift Score"},
    )
    fig.update_layout(template=PLOTLY_TEMPLATE, height=max(400, len(overview) * 22))
    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ── Distribution comparison ──
    st.subheader("Distribution Comparison")
    ctrl_col, chart_col = st.columns([1, 3])

    with ctrl_col:
        severity_filter = st.selectbox(
            "Filter by severity",
            options=["all", "high", "moderate", "low"],
            index=0, key="drift_sev_filter",
        )
        if severity_filter == "all":
            filtered = drift_df
        else:
            filtered = drift_df[drift_df["severity"] == severity_filter]

        feature_list = filtered["column"].tolist()
        if feature_list:
            selected_feature = st.selectbox("Select feature", feature_list, key="drift_feat")
            row = filtered[filtered["column"] == selected_feature].iloc[0]

            # Drift metric card
            sev = row["severity"]
            badge_cls = f"badge-{sev}"
            st.markdown(f'<span class="{badge_cls}">{sev.upper()}</span>', unsafe_allow_html=True)
            if row["feature_type"] == "numerical":
                st.markdown(f"**PSI:** {row.get('psi', 'N/A')}")
                st.markdown(f"**KS stat:** {row.get('ks_stat', 'N/A')}")
                st.markdown(f"**Wasserstein:** {row.get('wasserstein', 'N/A')}")
            else:
                st.markdown(f"**JSD:** {row.get('jsd', 'N/A')}")
                st.markdown(f"**Chi2 stat:** {row.get('chi2_stat', 'N/A')}")
        else:
            selected_feature = None

    with chart_col:
        if selected_feature is not None:
            ftype = row["feature_type"]
            if ftype == "numerical":
                train_vals = train[selected_feature].dropna()
                test_vals = test[selected_feature].dropna()
                fig = go.Figure()
                fig.add_trace(go.Histogram(
                    x=train_vals, name="Train", opacity=0.6,
                    histnorm="probability density", marker_color=TRAIN_COLOR,
                ))
                fig.add_trace(go.Histogram(
                    x=test_vals, name="Test", opacity=0.6,
                    histnorm="probability density", marker_color=TEST_COLOR,
                ))
                fig.update_layout(
                    barmode="overlay", template=PLOTLY_TEMPLATE,
                    title=f"{selected_feature} Distribution",
                    xaxis_title=selected_feature, yaxis_title="Density", height=450,
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                # Categorical / temporal: grouped bar
                train_counts = train[selected_feature].value_counts(normalize=True).reset_index()
                train_counts.columns = [selected_feature, "proportion"]
                train_counts["dataset"] = "Train"
                test_counts = test[selected_feature].value_counts(normalize=True).reset_index()
                test_counts.columns = [selected_feature, "proportion"]
                test_counts["dataset"] = "Test"
                combined = pd.concat([train_counts, test_counts], ignore_index=True)
                fig = px.bar(
                    combined, x=selected_feature, y="proportion", color="dataset",
                    barmode="group", title=f"{selected_feature} Distribution",
                    color_discrete_map={"Train": TRAIN_COLOR, "Test": TEST_COLOR},
                )
                fig.update_layout(template=PLOTLY_TEMPLATE, height=450)
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No features to display. Adjust the severity filter.")

    st.divider()

    # ── Monthly trends ──
    st.subheader("Feature Trend Over Time (Training Data)")
    temporal_cols = feature_types.get("temporal", [])
    if temporal_cols:
        month_col = temporal_cols[0]
        train_with_month = train.copy()
        train_with_month["_month_dt"] = pd.to_datetime(
            train_with_month[month_col], format="%y-%b", errors="coerce"
        )
        train_with_month = train_with_month.dropna(subset=["_month_dt"]).sort_values("_month_dt")

        trend_features = feature_types["numerical"] + feature_types["categorical"]
        trend_feature = st.selectbox("Select feature to track over time", trend_features, key="trend_feat")

        if trend_feature in feature_types["numerical"]:
            monthly = train_with_month.groupby(month_col, sort=False)[trend_feature].agg(
                ["mean", "std"]
            ).reset_index()
            monthly["_month_dt"] = pd.to_datetime(monthly[month_col], format="%y-%b", errors="coerce")
            monthly = monthly.sort_values("_month_dt")
            monthly["month_label"] = monthly["_month_dt"].dt.strftime("%b %y")

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=monthly["month_label"], y=monthly["mean"],
                mode="lines+markers", name="Mean", line=dict(color=TRAIN_COLOR),
            ))
            fig.add_trace(go.Scatter(
                x=monthly["month_label"].tolist() + monthly["month_label"].tolist()[::-1],
                y=(monthly["mean"] + monthly["std"]).tolist()
                + (monthly["mean"] - monthly["std"]).tolist()[::-1],
                fill="toself", fillcolor="rgba(99,110,250,0.15)",
                line=dict(color="rgba(255,255,255,0)"), name="Mean +/- Std",
            ))
            fig.update_layout(
                template=PLOTLY_TEMPLATE, height=400,
                title=f"{trend_feature} — Monthly Trend",
                xaxis_title="Month", yaxis_title=trend_feature,
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            ct = pd.crosstab(
                train_with_month[month_col], train_with_month[trend_feature], normalize="index"
            )
            ct["_month_dt"] = pd.to_datetime(ct.index, format="%y-%b", errors="coerce")
            ct = ct.sort_values("_month_dt")
            ct["month_label"] = ct["_month_dt"].dt.strftime("%b %y")
            ct = ct.drop(columns=["_month_dt"])
            melted = ct.melt(id_vars=["month_label"], var_name="category", value_name="proportion")
            fig = px.bar(
                melted, x="month_label", y="proportion", color="category",
                title=f"{trend_feature} — Monthly Proportions",
                labels={"month_label": "Month", "proportion": "Proportion"},
            )
            fig.update_layout(barmode="stack", template=PLOTLY_TEMPLATE, height=400)
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No temporal column detected — cannot show time trends.")

    # ── Full drift table ──
    with st.expander("Full Drift Table"):
        def color_severity(val):
            colors = {
                "high": "background-color: #ffcccc",
                "moderate": "background-color: #fff3cc",
                "low": "background-color: #ccffcc",
            }
            return colors.get(val, "")

        st.dataframe(
            drift_df.style.applymap(color_severity, subset=["severity"]),
            use_container_width=True, height=500,
        )

# ══════════════════════════════════════════════════════════════════════════
# TAB 3 : MITIGATION & MODEL COMPARISON
# ══════════════════════════════════════════════════════════════════════════
with tab_mitigation:
    st.header("Mitigation & Model Comparison")

    # ── Pipeline summary cards ──
    st.subheader("Mitigation Pipeline")
    p1, p2, p3 = st.columns(3)

    with p1:
        st.markdown("""<div class="pipeline-card">
            <h4>1. Normalize</h4>
            <p>City abbreviation mapping, lowercase + hyphen removal to fix data quality issues</p>
        </div>""", unsafe_allow_html=True)
    with p2:
        st.markdown("""<div class="pipeline-card">
            <h4>2. Detect Drift</h4>
            <p>Re-run drift detection on normalized data to measure real distributional shift (not case artifacts)</p>
        </div>""", unsafe_allow_html=True)
    with p3:
        n_dropped = len(drop_cols) if drop_cols else 0
        n_mitigated = len([e for e in mitigation_log if e["strategy_applied"] != "Dropped"]) if mitigation_log else 0
        st.markdown(f"""<div class="pipeline-card">
            <h4>3. Progressive Mitigation</h4>
            <p>Graduated strategies (clip, log, bin, rare-group, cyclical-encode) saved {n_mitigated} features; {n_dropped} dropped as last resort</p>
        </div>""", unsafe_allow_html=True)

    st.divider()

    # ── Data quality: before / after ──
    st.subheader("Data Quality Normalization: Before vs After")

    # Find categorical columns that were normalized (object dtype in raw data)
    skip_cols = set(feature_types.get("id", []) + feature_types.get("target", []))
    normalized_cols = [
        col for col in train.columns
        if col not in skip_cols and train[col].dtype == object
        and col not in feature_types.get("temporal", [])
    ]

    # Pick up to 2 example columns to show
    display_cols = normalized_cols[:2] if len(normalized_cols) >= 2 else normalized_cols

    if display_cols:
        cols = st.columns(len(display_cols))
        for i, col_name in enumerate(display_cols):
            with cols[i]:
                # Before
                before_train = train[col_name].value_counts().reset_index()
                before_train.columns = [col_name, "count"]
                before_train["dataset"] = "Train"
                before_test = test[col_name].value_counts().reset_index()
                before_test.columns = [col_name, "count"]
                before_test["dataset"] = "Test"
                before = pd.concat([before_train, before_test], ignore_index=True)

                # After
                after_train = train_m[col_name].value_counts().reset_index()
                after_train.columns = [col_name, "count"]
                after_train["dataset"] = "Train"
                after_test = test_m[col_name].value_counts().reset_index()
                after_test.columns = [col_name, "count"]
                after_test["dataset"] = "Test"
                after = pd.concat([after_train, after_test], ignore_index=True)

                fig = make_subplots(
                    rows=1, cols=2,
                    subplot_titles=["Before Normalization", "After Normalization"],
                    horizontal_spacing=0.12,
                )
                # Before traces
                for ds, color in [("Train", TRAIN_COLOR), ("Test", TEST_COLOR)]:
                    subset = before[before["dataset"] == ds]
                    fig.add_trace(go.Bar(
                        x=subset[col_name], y=subset["count"],
                        name=ds, marker_color=color,
                        legendgroup=ds, showlegend=(i == 0),
                    ), row=1, col=1)

                # After traces
                for ds, color in [("Train", TRAIN_COLOR), ("Test", TEST_COLOR)]:
                    subset = after[after["dataset"] == ds]
                    fig.add_trace(go.Bar(
                        x=subset[col_name], y=subset["count"],
                        name=ds, marker_color=color,
                        legendgroup=ds, showlegend=False,
                    ), row=1, col=2)

                fig.update_layout(
                    template=PLOTLY_TEMPLATE, height=400,
                    title_text=f"{col_name}", barmode="group",
                )
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No categorical columns were normalized.")

    st.divider()

    # ── Progressive mitigation summary ──
    st.subheader("Progressive Drift Mitigation")

    if mitigation_log:
        # Build mitigation actions table
        action_rows = []
        for entry in mitigation_log:
            action_rows.append({
                "Feature": entry["column"],
                "Type": entry["feature_type"],
                "Original Severity": entry["original_severity"].capitalize(),
                "Strategy": entry["strategy_applied"],
                "Final Severity": entry["final_severity"].capitalize(),
            })

        action_table, action_chart = st.columns([1, 1])
        with action_table:
            st.dataframe(pd.DataFrame(action_rows), use_container_width=True, hide_index=True)

        with action_chart:
            n_all_features = len([c for c in train.columns if c not in set(feature_types.get("id", []) + feature_types.get("target", []))])
            n_dropped = len(drop_cols)
            n_mitigated = len([e for e in mitigation_log if e["strategy_applied"] != "Dropped"])
            n_kept = n_all_features - n_dropped - n_mitigated
            labels = ["Untouched", "Mitigated", "Dropped"]
            values = [n_kept, n_mitigated, n_dropped]
            colors = ["#636EFA", "#00CC96", "#EF553B"]
            fig = go.Figure(go.Pie(
                labels=labels, values=values, hole=0.45,
                marker_colors=colors,
            ))
            fig.update_layout(
                template=PLOTLY_TEMPLATE, height=300,
                title=f"Features: {n_kept} untouched, {n_mitigated} mitigated, {n_dropped} dropped",
                margin=dict(t=40, b=20, l=20, r=20),
            )
            st.plotly_chart(fig, use_container_width=True)

        st.markdown(
            "High-drift features are progressively mitigated using graduated strategies "
            "(clipping, log transform, binning, rare-category grouping, cyclical encoding). "
            "Features are only **dropped as a last resort** when all strategies fail to reduce "
            "drift below the high-severity threshold."
        )
    elif drop_cols:
        st.dataframe(
            pd.DataFrame({"Dropped Features": drop_cols}),
            use_container_width=True, hide_index=True,
        )
    else:
        st.info("No high-drift features detected — no mitigation needed.")

    st.divider()

    # ── Model metrics comparison ──
    st.subheader("Model Performance Comparison")

    if baseline_metrics and mitigated_metrics:
        mc1, mc2, mc3, mc4 = st.columns(4)

        b_train = baseline_metrics.get("train_auprc")
        b_test = baseline_metrics.get("test_auprc")
        m_train = mitigated_metrics.get("train_auprc")
        m_test = mitigated_metrics.get("test_auprc")

        mc1.metric(
            "Baseline AU-PRC (Train)",
            f"{b_train:.4f}" if b_train else "N/A",
        )
        mc2.metric(
            "Baseline AU-PRC (Test)",
            f"{b_test:.4f}" if b_test else "N/A",
        )
        mc3.metric(
            "Mitigated AU-PRC (Train)",
            f"{m_train:.4f}" if m_train else "N/A",
        )
        mc4.metric(
            "Mitigated AU-PRC (Test)",
            f"{m_test:.4f}" if m_test else "N/A",
        )

        # Generalization gap
        if b_train and b_test and m_train and m_test:
            b_gap = b_train - b_test
            m_gap = m_train - m_test
            winner = "Baseline" if (b_test or 0) >= (m_test or 0) else "Mitigated"

            gc1, gc2, gc3 = st.columns(3)
            gc1.metric("Baseline Gap", f"{b_gap:.4f}")
            gc2.metric("Mitigated Gap", f"{m_gap:.4f}")
            gc3.metric("Best Test AU-PRC", winner)

        # Grouped bar comparison
        metrics_data = []
        for label, metrics in [("Baseline", baseline_metrics), ("Mitigated", mitigated_metrics)]:
            for key, name in [("train_auprc", "Train AU-PRC"), ("test_auprc", "Test AU-PRC")]:
                if key in metrics:
                    metrics_data.append({"Model": label, "Metric": name, "Value": metrics[key]})

        if metrics_data:
            fig = px.bar(
                pd.DataFrame(metrics_data),
                x="Metric", y="Value", color="Model", barmode="group",
                color_discrete_map={"Baseline": TRAIN_COLOR, "Mitigated": "#00CC96"},
                title="AU-PRC Comparison",
            )
            fig.update_layout(template=PLOTLY_TEMPLATE, height=400, yaxis_range=[0, 1])
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning(
            "Model files not found. Run `python src/main.py` first to generate "
            "`model_baseline.joblib` and `model_mitigated.joblib`."
        )

    st.divider()

    # ── Feature importance comparison ──
    st.subheader("Feature Importance Comparison")
    import joblib as _jl

    has_baseline = os.path.exists("model_baseline.joblib")
    has_mitigated = os.path.exists("model_mitigated.joblib")

    if has_baseline or has_mitigated:
        fi_cols = st.columns(2)

        if has_baseline:
            with fi_cols[0]:
                model_b = _jl.load("model_baseline.joblib")
                imp_b = pd.DataFrame({
                    "feature": model_b.feature_name_,
                    "importance": model_b.feature_importances_,
                }).sort_values("importance", ascending=True).tail(15)
                fig = px.bar(
                    imp_b, x="importance", y="feature", orientation="h",
                    title="Baseline — Top 15 Features",
                    color_discrete_sequence=[TRAIN_COLOR],
                )
                fig.update_layout(template=PLOTLY_TEMPLATE, height=450)
                st.plotly_chart(fig, use_container_width=True)

        if has_mitigated:
            with fi_cols[1]:
                model_m = _jl.load("model_mitigated.joblib")
                imp_m = pd.DataFrame({
                    "feature": model_m.feature_name_,
                    "importance": model_m.feature_importances_,
                }).sort_values("importance", ascending=True).tail(15)
                fig = px.bar(
                    imp_m, x="importance", y="feature", orientation="h",
                    title="Mitigated — Top 15 Features",
                    color_discrete_sequence=["#00CC96"],
                )
                fig.update_layout(template=PLOTLY_TEMPLATE, height=450)
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No model files found for feature importance.")

    st.divider()

    # ── Prediction distribution ──
    st.subheader("Prediction Score Distributions")

    if pred_baseline is not None and pred_mitigated is not None and target_col in test.columns:
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=["Baseline Predictions", "Mitigated Predictions"],
            horizontal_spacing=0.08,
        )

        id_col = feature_types["id"][0] if feature_types["id"] else "CustomerID"

        for col_idx, (pred_df, label) in enumerate(
            [(pred_baseline, "Baseline"), (pred_mitigated, "Mitigated")], start=1
        ):
            merged = pred_df.merge(test[[id_col, target_col]], left_on="CustomerID", right_on=id_col)
            for status, color, name in [
                ("No", TRAIN_COLOR, "No Churn"),
                ("Yes", TEST_COLOR, "Churn"),
            ]:
                subset = merged[merged[target_col] == status]
                fig.add_trace(go.Histogram(
                    x=subset["probability_score"],
                    name=name, opacity=0.6, marker_color=color,
                    legendgroup=name, showlegend=(col_idx == 1),
                    nbinsx=40,
                ), row=1, col=col_idx)

        fig.update_layout(
            template=PLOTLY_TEMPLATE, height=400, barmode="overlay",
            title_text="Predicted Probability by Actual Churn Status",
        )
        fig.update_xaxes(title_text="Probability Score")
        fig.update_yaxes(title_text="Count", col=1)
        st.plotly_chart(fig, use_container_width=True)
    elif pred_baseline is None and pred_mitigated is None:
        st.warning("No prediction files found. Run the pipeline first.")
    else:
        st.info("Test labels required for prediction distribution breakdown.")
