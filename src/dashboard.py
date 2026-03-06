import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from profiler import load_data, detect_feature_types
from drift_detector import run_drift_analysis

st.set_page_config(page_title="NAISC Drift Dashboard", layout="wide")
st.title("NAISC Drift Detection Dashboard")

# --- Sidebar: data paths ---
st.sidebar.header("Data Sources")
default_train = "public_data/train.csv"
default_test = "public_data/test.csv"
train_path = st.sidebar.text_input("Train CSV path", value=default_train)
test_path = st.sidebar.text_input("Test CSV path", value=default_test)


@st.cache_data
def load_and_analyse(train_path: str, test_path: str):
    train, test = load_data(train_path, test_path)
    feature_types = detect_feature_types(train)
    drift_df = run_drift_analysis(train, test, feature_types)
    return train, test, feature_types, drift_df


if not os.path.exists(train_path) or not os.path.exists(test_path):
    st.error("Please provide valid file paths in the sidebar.")
    st.stop()

train, test, feature_types, drift_df = load_and_analyse(train_path, test_path)

# --- Drift Summary Table ---
st.header("Drift Summary")

severity_filter = st.multiselect(
    "Filter by severity",
    options=["high", "moderate", "low"],
    default=["high", "moderate"],
)
filtered = drift_df[drift_df["severity"].isin(severity_filter)]


def color_severity(val):
    colors = {"high": "background-color: #ffcccc", "moderate": "background-color: #fff3cc", "low": "background-color: #ccffcc"}
    return colors.get(val, "")


st.dataframe(
    filtered.style.applymap(color_severity, subset=["severity"]),
    use_container_width=True,
    height=400,
)

st.markdown(f"**{len(filtered)}** features shown | "
            f"{(drift_df['severity'] == 'high').sum()} high, "
            f"{(drift_df['severity'] == 'moderate').sum()} moderate, "
            f"{(drift_df['severity'] == 'low').sum()} low")

# --- Distribution Plots ---
st.header("Distribution Comparison (Train vs Test)")

plot_features = filtered["column"].tolist()
if not plot_features:
    st.info("No features to plot. Adjust the severity filter above.")
    st.stop()

selected_feature = st.selectbox("Select feature", plot_features)
row = filtered[filtered["column"] == selected_feature].iloc[0]
ftype = row["feature_type"]

if ftype == "numerical":
    # Overlaid histograms
    train_vals = train[selected_feature].dropna()
    test_vals = test[selected_feature].dropna()

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=train_vals, name="Train", opacity=0.6,
        histnorm="probability density",
        marker_color="#636EFA",
    ))
    fig.add_trace(go.Histogram(
        x=test_vals, name="Test", opacity=0.6,
        histnorm="probability density",
        marker_color="#EF553B",
    ))
    fig.update_layout(
        barmode="overlay",
        title=f"{selected_feature} — KS={row['ks_stat']:.4f}, PSI={row['psi']:.4f}",
        xaxis_title=selected_feature,
        yaxis_title="Density",
        height=450,
    )
    st.plotly_chart(fig, use_container_width=True)

else:
    # Categorical: grouped bar chart with proportions
    train_counts = train[selected_feature].value_counts(normalize=True).reset_index()
    train_counts.columns = [selected_feature, "proportion"]
    train_counts["dataset"] = "Train"

    test_counts = test[selected_feature].value_counts(normalize=True).reset_index()
    test_counts.columns = [selected_feature, "proportion"]
    test_counts["dataset"] = "Test"

    combined = pd.concat([train_counts, test_counts], ignore_index=True)

    jsd_val = row.get("jsd", 0)
    fig = px.bar(
        combined,
        x=selected_feature,
        y="proportion",
        color="dataset",
        barmode="group",
        title=f"{selected_feature} — JSD={jsd_val:.4f}",
        color_discrete_map={"Train": "#636EFA", "Test": "#EF553B"},
        height=450,
    )
    st.plotly_chart(fig, use_container_width=True)

# --- Temporal Drift: Feature over time ---
st.header("Feature Trend Over Time (Intra-Training)")

temporal_cols = feature_types.get("temporal", [])
if temporal_cols:
    month_col = temporal_cols[0]
    # Parse month to datetime for proper ordering
    train_with_month = train.copy()
    train_with_month["_month_dt"] = pd.to_datetime(train_with_month[month_col], format="%y-%b", errors="coerce")
    train_with_month = train_with_month.dropna(subset=["_month_dt"]).sort_values("_month_dt")
    month_order = train_with_month["_month_dt"].dt.strftime("%b").unique().tolist()

    # Pick features to plot (exclude ID, target, temporal)
    trend_features = feature_types["numerical"] + feature_types["categorical"]
    trend_feature = st.selectbox("Select feature to track over time", trend_features, key="trend_feature")

    if trend_feature in feature_types["numerical"]:
        # Line chart: mean and median per month
        monthly = train_with_month.groupby(month_col, sort=False)[trend_feature].agg(["mean", "median", "std"]).reset_index()
        monthly["_month_dt"] = pd.to_datetime(monthly[month_col], format="%y-%b", errors="coerce")
        monthly = monthly.sort_values("_month_dt")
        monthly["month_label"] = monthly["_month_dt"].dt.strftime("%b %y")

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=monthly["month_label"], y=monthly["mean"],
            mode="lines+markers", name="Mean",
            line=dict(color="#636EFA"),
        ))
        fig.add_trace(go.Scatter(
            x=monthly["month_label"], y=monthly["median"],
            mode="lines+markers", name="Median",
            line=dict(color="#00CC96", dash="dash"),
        ))
        # Std band around mean
        fig.add_trace(go.Scatter(
            x=monthly["month_label"].tolist() + monthly["month_label"].tolist()[::-1],
            y=(monthly["mean"] + monthly["std"]).tolist() + (monthly["mean"] - monthly["std"]).tolist()[::-1],
            fill="toself", fillcolor="rgba(99,110,250,0.15)",
            line=dict(color="rgba(255,255,255,0)"),
            name="Mean ± Std",
        ))
        fig.update_layout(
            title=f"{trend_feature} — Monthly Trend (Train)",
            xaxis_title="Month", yaxis_title=trend_feature,
            height=450,
        )
        st.plotly_chart(fig, use_container_width=True)

    else:
        # Stacked bar: category proportions per month
        ct = pd.crosstab(train_with_month[month_col], train_with_month[trend_feature], normalize="index")
        ct["_month_dt"] = pd.to_datetime(ct.index, format="%y-%b", errors="coerce")
        ct = ct.sort_values("_month_dt")
        ct["month_label"] = ct["_month_dt"].dt.strftime("%b %y")
        ct = ct.drop(columns=["_month_dt"])

        melted = ct.melt(id_vars=["month_label"], var_name="category", value_name="proportion")

        fig = px.bar(
            melted, x="month_label", y="proportion", color="category",
            title=f"{trend_feature} — Monthly Proportions (Train)",
            labels={"month_label": "Month", "proportion": "Proportion"},
            height=450,
        )
        fig.update_layout(barmode="stack")
        st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No temporal column detected — cannot show time trends.")

# --- All drifted features overview ---
st.header("Drift Severity Overview")

severity_order = {"high": 3, "moderate": 2, "low": 1}
overview = drift_df.copy()
overview["severity_rank"] = overview["severity"].map(severity_order)
overview = overview.sort_values("severity_rank", ascending=False)

# Pick the primary metric per feature for a bar chart
def get_drift_score(row):
    if row["feature_type"] == "numerical":
        return row.get("psi", 0) or 0
    return row.get("jsd", 0) or 0

overview["drift_score"] = overview.apply(get_drift_score, axis=1)

fig = px.bar(
    overview,
    x="column",
    y="drift_score",
    color="severity",
    color_discrete_map={"high": "#EF553B", "moderate": "#FECB52", "low": "#00CC96"},
    title="Drift Score by Feature (PSI for numerical, JSD for categorical)",
    labels={"column": "Feature", "drift_score": "Drift Score"},
    height=500,
)
fig.update_layout(xaxis_tickangle=-45)
st.plotly_chart(fig, use_container_width=True)

# --- Model performance (if model.joblib exists) ---
if os.path.exists("model.joblib"):
    import joblib
    from sklearn.metrics import average_precision_score

    st.header("Model Performance")

    model = joblib.load("model.joblib")
    target_col = feature_types["target"][0] if feature_types["target"] else "ChurnStatus"

    importances = pd.DataFrame({
        "feature": model.feature_name_,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False).head(20)

    fig = px.bar(
        importances,
        x="importance",
        y="feature",
        orientation="h",
        title="Top 20 Feature Importances (Gain)",
        height=500,
    )
    fig.update_layout(yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig, use_container_width=True)
