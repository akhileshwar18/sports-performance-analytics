# ============================================================
# SPORTS PERFORMANCE ANALYTICS — STREAMLIT APP
# K-Means -> Random Forest -> Linear Regression pipeline
# ============================================================

import json
import pickle

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ------------------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------------------
st.set_page_config(
    page_title="Athlete Performance Analytics",
    page_icon="🏅",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODEL_DIR = "models"


# ------------------------------------------------------------
# LOAD ARTIFACTS (cached so the app stays fast)
# ------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    with open(f"{MODEL_DIR}/cluster_scaler.pkl", "rb") as f:
        cluster_scaler = pickle.load(f)
    with open(f"{MODEL_DIR}/kmeans_model.pkl", "rb") as f:
        kmeans_model = pickle.load(f)
    with open(f"{MODEL_DIR}/rf_scaler.pkl", "rb") as f:
        rf_scaler = pickle.load(f)
    with open(f"{MODEL_DIR}/rf_model.pkl", "rb") as f:
        rf_model = pickle.load(f)
    with open(f"{MODEL_DIR}/lr_model.pkl", "rb") as f:
        lr_model = pickle.load(f)
    with open(f"{MODEL_DIR}/feature_medians.pkl", "rb") as f:
        feature_medians = pickle.load(f)
    with open(f"{MODEL_DIR}/cluster_labels.pkl", "rb") as f:
        cluster_labels = pickle.load(f)
    with open(f"{MODEL_DIR}/meta.json", "r") as f:
        meta = json.load(f)
    return (
        cluster_scaler,
        kmeans_model,
        rf_scaler,
        rf_model,
        lr_model,
        feature_medians,
        cluster_labels,
        meta,
    )


@st.cache_data
def load_dataset():
    return pd.read_csv(f"{MODEL_DIR}/athlete_dataset_scored.csv")


(
    cluster_scaler,
    kmeans_model,
    rf_scaler,
    rf_model,
    lr_model,
    feature_medians,
    cluster_labels,
    meta,
) = load_artifacts()

data = load_dataset()

FEATURES = meta["FEATURES"]
RF_FEATURES = meta["RF_FEATURES"]
LR_FEATURES = meta["LR_FEATURES"]

FEATURE_META = {
    "Age": ("Age (years)", 14, 45, 24, 1),
    "Weight_kg": ("Weight (kg)", 40.0, 130.0, 72.0, 0.1),
    "Training_Experience_Years": ("Training Experience (yrs)", 0, 25, 4, 1),
    "Training_Hours_Per_Week": ("Training Hours / Week", 0.0, 30.0, 10.0, 0.1),
    "VO2_Max_ml_kg_min": ("VO2 Max (ml/kg/min)", 25.0, 85.0, 50.0, 0.1),
    "Sprint_Speed_m_s": ("Sprint Speed (m/s)", 4.0, 12.0, 8.0, 0.01),
    "Reaction_Time_ms": ("Reaction Time (ms)", 120, 400, 220, 1),
    "Strength_Test_Score": ("Strength Test Score", 30.0, 100.0, 75.0, 0.1),
    "Resting_Heart_Rate_bpm": ("Resting Heart Rate (bpm)", 40, 100, 65, 1),
    "Sleep_Hours_Per_Day": ("Sleep Hours / Day", 3.0, 12.0, 7.5, 0.1),
    "Training_Days_Per_Week": ("Training Days / Week", 1, 7, 5, 1),
    "Recovery_Heart_Rate_bpm": ("Recovery Heart Rate (bpm)", 50, 140, 90, 1),
}


def get_recommendation(row: dict) -> str:
    if row["Predicted_Overall_Performance_%"] >= 75:
        return "Good performance — maintain current training."

    recs = []
    checks = [
        ("Training_Hours_Per_Week", "<", "Improve training consistency."),
        ("VO2_Max_ml_kg_min", "<", "Improve aerobic fitness."),
        ("Strength_Test_Score", "<", "Improve strength training."),
        ("Sprint_Speed_m_s", "<", "Improve speed training."),
        ("Reaction_Time_ms", ">", "Practice reaction-time drills."),
        ("Sleep_Hours_Per_Day", "<", "Improve sleep and recovery."),
        ("Training_Days_Per_Week", "<", "Increase training consistency."),
        ("Resting_Heart_Rate_bpm", ">", "Focus on cardiovascular fitness."),
        ("Recovery_Heart_Rate_bpm", ">", "Improve conditioning and recovery."),
    ]
    for feat, op, msg in checks:
        med = feature_medians[feat]
        if (op == "<" and row[feat] < med) or (op == ">" and row[feat] > med):
            recs.append(msg)

    if not recs:
        return "Performance is below 75%. Maintain balanced training and monitor progress."
    return "; ".join(recs)


def predict_athlete(input_dict: dict):
    """Run the full 3-stage pipeline on one athlete's raw features."""
    row = pd.DataFrame([input_dict])[FEATURES]

    # Stage 1 — K-Means
    scaled = cluster_scaler.transform(row)
    cluster = int(kmeans_model.predict(scaled)[0])
    level = cluster_labels.get(cluster, cluster_labels.get(str(cluster), f"Cluster {cluster}"))

    # Stage 2 — Random Forest
    row_rf = row.copy()
    row_rf["Cluster"] = cluster
    row_rf = row_rf[RF_FEATURES]
    scaled_rf = rf_scaler.transform(row_rf)
    perf = float(np.clip(rf_model.predict(scaled_rf)[0], 0, 100))

    # Stage 3 — Linear Regression
    row_lr = pd.DataFrame([{
        "Age": input_dict["Age"],
        "Weight_kg": input_dict["Weight_kg"],
        "Training_Days_Per_Week": input_dict["Training_Days_Per_Week"],
        "Sleep_Hours_Per_Day": input_dict["Sleep_Hours_Per_Day"],
        "Cluster": cluster,
        "Predicted_Overall_Performance_%": perf,
    }])[LR_FEATURES]
    medal = float(np.clip(lr_model.predict(row_lr)[0], 0, 100))

    rec_input = dict(input_dict)
    rec_input["Predicted_Overall_Performance_%"] = perf
    recommendation = get_recommendation(rec_input)

    return {
        "cluster": cluster,
        "performance_level": level,
        "predicted_performance": round(perf, 2),
        "predicted_medal_chance": round(medal, 2),
        "recommendation": recommendation,
    }


# ------------------------------------------------------------
# STYLE
# ------------------------------------------------------------
st.markdown(
    """
    <style>
    .metric-card {
        background: linear-gradient(135deg, #1e3a5f 0%, #2c5282 100%);
        padding: 1.2rem 1.5rem; border-radius: 12px; color: white;
    }
    .rec-box {
        background-color: #0f2b1d; border-left: 4px solid #2ecc71;
        padding: 1rem 1.2rem; border-radius: 6px; margin-top: 0.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------------
# SIDEBAR NAVIGATION
# ------------------------------------------------------------
st.sidebar.title("🏅 Athlete Analytics")
page = st.sidebar.radio(
    "Navigate",
    ["🏠 Overview", "🔮 Predict New Athlete", "👥 Athlete Explorer", "📊 Model Performance"],
)
st.sidebar.markdown("---")
st.sidebar.caption(
    f"Pipeline: K-Means (K={meta['best_k']}) → Random Forest → Linear Regression"
)
st.sidebar.caption(f"Trained on {meta['n_athletes']:,} athletes")

# ============================================================
# PAGE: OVERVIEW
# ============================================================
if page == "🏠 Overview":
    st.title("🏅 Athlete Performance Analytics")
    st.caption(
        "A 3-stage ML pipeline — clustering, performance prediction, and medal-chance "
        "estimation — trained on athlete training and physiological data."
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Athletes in Dataset", f"{len(data):,}")
    c2.metric("Performance Clusters", meta["best_k"])
    c3.metric("RF R² (Performance)", meta["rf_metrics"]["R2"])
    c4.metric("LR R² (Medal Chance)", meta["lr_metrics"]["R2"])

    st.markdown("### How the pipeline works")
    st.markdown(
        """
        1. **K-Means clustering** groups athletes into performance tiers using 12 training
           and physiological features (best K chosen by silhouette score).
        2. **Random Forest regression** predicts each athlete's *Overall Performance Score*
           using those same features plus their cluster.
        3. **Linear Regression** takes the predicted performance score — along with age,
           weight, training days, sleep, and cluster — to estimate *Medal Chance %*.
        4. A rule-based engine compares an athlete's stats to the dataset medians and
           surfaces targeted training recommendations.
        """
    )

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Performance Score distribution")
        fig = px.histogram(
            data, x="Overall_Performance_Score_%", nbins=40,
            color_discrete_sequence=["#2c5282"],
        )
        fig.update_layout(height=350, margin=dict(t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.markdown("#### Athletes per performance tier")
        counts = data["Performance_Level"].value_counts().reset_index()
        counts.columns = ["Performance_Level", "Count"]
        fig = px.pie(
            counts, names="Performance_Level", values="Count", hole=0.45,
            color_discrete_sequence=px.colors.sequential.Blues_r,
        )
        fig.update_layout(height=350, margin=dict(t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

# ============================================================
# PAGE: PREDICT NEW ATHLETE
# ============================================================
elif page == "🔮 Predict New Athlete":
    st.title("🔮 Predict a New Athlete")
    st.caption("Enter an athlete's stats to get their performance tier, predicted score, and medal chance.")

    with st.form("athlete_form"):
        athlete_id = st.text_input("Athlete ID / Name", value="A-NEW-001")

        cols = st.columns(3)
        input_dict = {}
        for i, feat in enumerate(FEATURES):
            label, lo, hi, default, step = FEATURE_META[feat]
            with cols[i % 3]:
                input_dict[feat] = st.number_input(
                    label, min_value=float(lo), max_value=float(hi),
                    value=float(default), step=float(step), key=feat,
                )

        submitted = st.form_submit_button("Run Prediction", type="primary", use_container_width=True)

    if submitted:
        result = predict_athlete(input_dict)

        st.markdown("### Results")
        c1, c2, c3 = st.columns(3)
        c1.metric("Performance Tier", result["performance_level"])
        c2.metric("Predicted Overall Performance", f"{result['predicted_performance']}%")
        c3.metric("Predicted Medal Chance", f"{result['predicted_medal_chance']}%")

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=result["predicted_medal_chance"],
            title={"text": "Medal Chance %"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#2c5282"},
                "steps": [
                    {"range": [0, 33], "color": "#fde2e2"},
                    {"range": [33, 66], "color": "#fff3cd"},
                    {"range": [66, 100], "color": "#d4edda"},
                ],
            },
        ))
        fig.update_layout(height=280, margin=dict(t=40, b=10))
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### Training Recommendation")
        st.markdown(f'<div class="rec-box">{result["recommendation"]}</div>', unsafe_allow_html=True)

        with st.expander("Add this athlete to the working dataset (session only)"):
            if st.button("Add to session table"):
                new_row = dict(input_dict)
                new_row["Athlete_ID"] = athlete_id
                new_row["Cluster"] = result["cluster"]
                new_row["Performance_Level"] = result["performance_level"]
                new_row["Predicted_Overall_Performance_%"] = result["predicted_performance"]
                new_row["Predicted_Medal_Chance%"] = result["predicted_medal_chance"]
                new_row["Recommendation"] = result["recommendation"]
                if "session_athletes" not in st.session_state:
                    st.session_state.session_athletes = []
                st.session_state.session_athletes.append(new_row)
                st.success("Added — view it in Athlete Explorer > Session Additions.")

# ============================================================
# PAGE: ATHLETE EXPLORER
# ============================================================
elif page == "👥 Athlete Explorer":
    st.title("👥 Athlete Explorer")

    tab1, tab2 = st.tabs(["Full Dataset", "Session Additions"])

    with tab1:
        f1, f2, f3 = st.columns(3)
        with f1:
            tiers = st.multiselect(
                "Performance tier", options=sorted(data["Performance_Level"].unique()),
                default=list(sorted(data["Performance_Level"].unique())),
            )
        with f2:
            min_perf, max_perf = st.slider(
                "Overall Performance %", 0, 100, (0, 100)
            )
        with f3:
            search_id = st.text_input("Search Athlete ID contains")

        filtered = data[
            data["Performance_Level"].isin(tiers)
            & data["Overall_Performance_Score_%"].between(min_perf, max_perf)
        ]
        if search_id:
            filtered = filtered[filtered["Athlete_ID"].astype(str).str.contains(search_id, case=False)]

        st.caption(f"{len(filtered):,} athletes match your filters")
        st.dataframe(
            filtered[
                ["Athlete_ID", "Performance_Level", "Overall_Performance_Score_%",
                 "Predicted_Overall_Performance_%", "Medal_Chance%",
                 "Predicted_Medal_Chance%", "Recommendation"]
            ],
            use_container_width=True, height=420,
        )
        st.download_button(
            "Download filtered results as CSV",
            filtered.to_csv(index=False).encode("utf-8"),
            file_name="filtered_athletes.csv", mime="text/csv",
        )

    with tab2:
        session_athletes = st.session_state.get("session_athletes", [])
        if not session_athletes:
            st.info("No athletes added yet — use the Predict page to add some.")
        else:
            sdf = pd.DataFrame(session_athletes)
            st.dataframe(sdf, use_container_width=True)
            st.download_button(
                "Download session athletes as CSV",
                sdf.to_csv(index=False).encode("utf-8"),
                file_name="session_athletes.csv", mime="text/csv",
            )

# ============================================================
# PAGE: MODEL PERFORMANCE
# ============================================================
elif page == "📊 Model Performance":
    st.title("📊 Model Performance")

    st.markdown("### K-Means — choosing K by silhouette score")
    sil = meta["silhouette_scores"]
    sil_df = pd.DataFrame({"K": list(sil.keys()), "Silhouette Score": list(sil.values())})
    fig = px.line(sil_df, x="K", y="Silhouette Score", markers=True)
    fig.add_vline(x=str(meta["best_k"]), line_dash="dash", line_color="green")
    fig.update_layout(height=320, margin=dict(t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)
    st.caption(f"Best K = {meta['best_k']} (highest silhouette score)")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Random Forest — Overall Performance")
        m = meta["rf_metrics"]
        st.metric("MAE", m["MAE"])
        st.metric("RMSE", m["RMSE"])
        st.metric("R²", m["R2"])
    with col2:
        st.markdown("### Linear Regression — Medal Chance")
        m = meta["lr_metrics"]
        st.metric("MAE", m["MAE"])
        st.metric("RMSE", m["RMSE"])
        st.metric("R²", m["R2"])

    st.markdown("### Random Forest — Feature Importance")
    imp_df = pd.DataFrame(meta["feature_importance"]).sort_values("Importance")
    fig = px.bar(imp_df, x="Importance", y="Feature", orientation="h",
                 color="Importance", color_continuous_scale="Blues")
    fig.update_layout(height=420, margin=dict(t=10, b=10), coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Actual vs. Predicted (Overall Performance)")
    fig = px.scatter(
        data, x="Overall_Performance_Score_%", y="Predicted_Overall_Performance_%",
        opacity=0.4, color_discrete_sequence=["#2c5282"],
    )
    lims = [data["Overall_Performance_Score_%"].min(), data["Overall_Performance_Score_%"].max()]
    fig.add_trace(go.Scatter(x=lims, y=lims, mode="lines", line=dict(color="red", dash="dash"), name="Perfect fit"))
    fig.update_layout(height=420, margin=dict(t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.caption("Built with Streamlit · scikit-learn · pickle")
