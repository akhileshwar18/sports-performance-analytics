# ============================================================
# TRAINING SCRIPT
# Runs the 3-algorithm pipeline (K-Means -> Random Forest ->
# Linear Regression) and pickles every artifact the Streamlit
# app needs at inference time.
#
# Run once, locally or in CI, whenever the source CSV changes:
#   python train_and_pickle.py
# ============================================================

import json
import pickle
import warnings

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    silhouette_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore")

DATA_PATH = "Sports_analytics_file.csv"
MODEL_DIR = "models"

FEATURES = [
    "Age",
    "Weight_kg",
    "Training_Experience_Years",
    "Training_Hours_Per_Week",
    "VO2_Max_ml_kg_min",
    "Sprint_Speed_m_s",
    "Reaction_Time_ms",
    "Strength_Test_Score",
    "Resting_Heart_Rate_bpm",
    "Sleep_Hours_Per_Day",
    "Training_Days_Per_Week",
    "Recovery_Heart_Rate_bpm",
]

# ------------------------------------------------------------
# 1. LOAD + CLEAN
# ------------------------------------------------------------
df = pd.read_csv(DATA_PATH)
data = df.copy()

for col in FEATURES:
    data[col] = pd.to_numeric(data[col], errors="coerce")
data["Overall_Performance_Score_%"] = pd.to_numeric(
    data["Overall_Performance_Score_%"], errors="coerce"
)
data["Medal_Chance%"] = pd.to_numeric(data["Medal_Chance%"], errors="coerce")

data = data.dropna(
    subset=FEATURES + ["Overall_Performance_Score_%", "Medal_Chance%"]
).reset_index(drop=True)

print(f"Loaded {len(data)} athlete rows after cleaning.")

# ------------------------------------------------------------
# 2. ALGORITHM 1 - K-MEANS
# ------------------------------------------------------------
X_cluster = data[FEATURES].copy()
cluster_scaler = MinMaxScaler()
X_cluster_scaled = cluster_scaler.fit_transform(X_cluster)

k_values = list(range(2, 7))
silhouette_scores = []
for k in k_values:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_cluster_scaled)
    silhouette_scores.append(silhouette_score(X_cluster_scaled, labels))
    print(f"K={k}  silhouette={silhouette_scores[-1]:.3f}")

best_k = k_values[int(np.argmax(silhouette_scores))]
print("Best K:", best_k)

kmeans_model = KMeans(n_clusters=best_k, random_state=42, n_init=10)
data["Cluster"] = kmeans_model.fit_predict(X_cluster_scaled)

cluster_summary = data.groupby("Cluster")[
    FEATURES + ["Overall_Performance_Score_%", "Medal_Chance%"]
].mean()

cluster_order = (
    cluster_summary["Overall_Performance_Score_%"].sort_values(ascending=False).index
)
if best_k == 3:
    names = ["High Performer", "Intermediate", "Developing"]
    cluster_labels = {c: names[i] for i, c in enumerate(cluster_order)}
else:
    cluster_labels = {
        c: f"Performance Group {i + 1}" for i, c in enumerate(cluster_order)
    }
data["Performance_Level"] = data["Cluster"].map(cluster_labels)

# ------------------------------------------------------------
# 3. ALGORITHM 2 - RANDOM FOREST (predict Overall Performance)
# ------------------------------------------------------------
RF_FEATURES = FEATURES + ["Cluster"]
X_rf = data[RF_FEATURES]
y_rf = data["Overall_Performance_Score_%"]

X_train_rf, X_test_rf, y_train_rf, y_test_rf = train_test_split(
    X_rf, y_rf, test_size=0.40, random_state=42
)

rf_scaler = MinMaxScaler()
X_train_rf_s = rf_scaler.fit_transform(X_train_rf)
X_test_rf_s = rf_scaler.transform(X_test_rf)

rf_model = RandomForestRegressor(
    n_estimators=4,
    max_depth=2,
    min_samples_split=100,
    min_samples_leaf=60,
    max_features=0.1,
    random_state=42,
    n_jobs=-1,
)
rf_model.fit(X_train_rf_s, y_train_rf)

rf_pred = rf_model.predict(X_test_rf_s)
rf_metrics = {
    "MAE": round(mean_absolute_error(y_test_rf, rf_pred), 2),
    "RMSE": round(float(np.sqrt(mean_squared_error(y_test_rf, rf_pred))), 2),
    "R2": round(r2_score(y_test_rf, rf_pred), 3),
}
print("Random Forest metrics:", rf_metrics)

X_rf_full = rf_scaler.transform(data[RF_FEATURES])
data["Predicted_Overall_Performance_%"] = (
    rf_model.predict(X_rf_full).clip(0, 100).round(2)
)

importance = (
    pd.DataFrame(
        {"Feature": RF_FEATURES, "Importance": rf_model.feature_importances_}
    )
    .sort_values("Importance", ascending=False)
    .reset_index(drop=True)
)

# ------------------------------------------------------------
# 4. ALGORITHM 3 - LINEAR REGRESSION (predict Medal Chance)
# ------------------------------------------------------------
LR_FEATURES = [
    "Age",
    "Weight_kg",
    "Training_Days_Per_Week",
    "Sleep_Hours_Per_Day",
    "Cluster",
    "Predicted_Overall_Performance_%",
]
X_lr = data[LR_FEATURES]
y_lr = data["Medal_Chance%"]

X_train_lr, X_test_lr, y_train_lr, y_test_lr = train_test_split(
    X_lr, y_lr, test_size=0.40, random_state=42
)

lr_model = LinearRegression()
lr_model.fit(X_train_lr, y_train_lr)

lr_pred = lr_model.predict(X_test_lr)
lr_metrics = {
    "MAE": round(mean_absolute_error(y_test_lr, lr_pred), 3),
    "RMSE": round(float(np.sqrt(mean_squared_error(y_test_lr, lr_pred))), 3),
    "R2": round(r2_score(y_test_lr, lr_pred), 3),
}
print("Linear Regression metrics:", lr_metrics)

data["Predicted_Medal_Chance%"] = (
    lr_model.predict(data[LR_FEATURES]).clip(0, 100).round(2)
)

# ------------------------------------------------------------
# 5. RECOMMENDATION ENGINE (rule-based, uses feature medians)
# ------------------------------------------------------------
feature_medians = data[FEATURES].median()


def get_recommendation(row):
    if row["Predicted_Overall_Performance_%"] >= 75:
        return "Good performance - maintain current training."

    recs = []
    if row["Training_Hours_Per_Week"] < feature_medians["Training_Hours_Per_Week"]:
        recs.append("Improve training consistency.")
    if row["VO2_Max_ml_kg_min"] < feature_medians["VO2_Max_ml_kg_min"]:
        recs.append("Improve aerobic fitness.")
    if row["Strength_Test_Score"] < feature_medians["Strength_Test_Score"]:
        recs.append("Improve strength training.")
    if row["Sprint_Speed_m_s"] < feature_medians["Sprint_Speed_m_s"]:
        recs.append("Improve speed training.")
    if row["Reaction_Time_ms"] > feature_medians["Reaction_Time_ms"]:
        recs.append("Practice reaction-time drills.")
    if row["Sleep_Hours_Per_Day"] < feature_medians["Sleep_Hours_Per_Day"]:
        recs.append("Improve sleep and recovery.")
    if row["Training_Days_Per_Week"] < feature_medians["Training_Days_Per_Week"]:
        recs.append("Increase training consistency.")
    if row["Resting_Heart_Rate_bpm"] > feature_medians["Resting_Heart_Rate_bpm"]:
        recs.append("Focus on cardiovascular fitness.")
    if row["Recovery_Heart_Rate_bpm"] > feature_medians["Recovery_Heart_Rate_bpm"]:
        recs.append("Improve conditioning and recovery.")

    if not recs:
        return "Performance is below 75%. Maintain balanced training and monitor progress."
    return "; ".join(recs)


data["Recommendation"] = data.apply(get_recommendation, axis=1)

# ------------------------------------------------------------
# 6. SAVE EVERYTHING THE APP NEEDS
# ------------------------------------------------------------
import os

os.makedirs(MODEL_DIR, exist_ok=True)

with open(f"{MODEL_DIR}/cluster_scaler.pkl", "wb") as f:
    pickle.dump(cluster_scaler, f)
with open(f"{MODEL_DIR}/kmeans_model.pkl", "wb") as f:
    pickle.dump(kmeans_model, f)
with open(f"{MODEL_DIR}/rf_scaler.pkl", "wb") as f:
    pickle.dump(rf_scaler, f)
with open(f"{MODEL_DIR}/rf_model.pkl", "wb") as f:
    pickle.dump(rf_model, f)
with open(f"{MODEL_DIR}/lr_model.pkl", "wb") as f:
    pickle.dump(lr_model, f)
with open(f"{MODEL_DIR}/feature_medians.pkl", "wb") as f:
    pickle.dump(feature_medians, f)
with open(f"{MODEL_DIR}/cluster_labels.pkl", "wb") as f:
    pickle.dump(cluster_labels, f)

meta = {
    "FEATURES": FEATURES,
    "RF_FEATURES": RF_FEATURES,
    "LR_FEATURES": LR_FEATURES,
    "best_k": int(best_k),
    "silhouette_scores": {str(k): float(s) for k, s in zip(k_values, silhouette_scores)},
    "rf_metrics": rf_metrics,
    "lr_metrics": lr_metrics,
    "feature_importance": importance.to_dict(orient="records"),
    "n_athletes": int(len(data)),
}
with open(f"{MODEL_DIR}/meta.json", "w") as f:
    json.dump(meta, f, indent=2)

# Save the enriched dataset (with clusters/predictions) for the app's
# "athlete database" browsing / leaderboard views.
data.to_csv(f"{MODEL_DIR}/athlete_dataset_scored.csv", index=False)

print("\nAll artifacts saved to ./models/")
print(os.listdir(MODEL_DIR))
