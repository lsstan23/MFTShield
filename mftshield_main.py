import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

st.set_page_config(layout="wide")
st.title("🚨 Real-Time Anomaly Detection in MFT Logs")

@st.cache_data
def load_data():
    n_samples = 500
    anomalies = 10
    np.random.seed(42)
    data = {
        "file_size_MB": np.concatenate([np.random.normal(5, 1, n_samples), np.random.normal(50, 5, anomalies)]),
        "transfer_time_sec": np.concatenate([np.random.normal(10, 2, n_samples), np.random.normal(100, 10, anomalies)]),
        "hour_of_day": np.concatenate([np.random.normal(14, 3, n_samples), np.random.normal(3, 1, anomalies)]),
        "unique_dest_ips": np.concatenate([np.random.normal(1, 0.5, n_samples), np.random.normal(5, 1, anomalies)]),
        "user_id_encoded": np.concatenate([np.random.normal(1000, 50, n_samples), np.random.normal(5000, 100, anomalies)]),
    }
    df = pd.DataFrame(data)
    df["label"] = [0] * n_samples + [1] * anomalies
    return df

df = load_data()
model = IsolationForest(contamination=0.02, random_state=42)
df["anomaly_score"] = model.fit_predict(df.drop(columns=["label"]))
df["detected_anomaly"] = df["anomaly_score"].apply(lambda x: 1 if x == -1 else 0)

# Metrics
true_positives = sum((df["label"] == 1) & (df["detected_anomaly"] == 1))
false_positives = sum((df["label"] == 0) & (df["detected_anomaly"] == 1))
false_negatives = sum((df["label"] == 1) & (df["detected_anomaly"] == 0))
precision = true_positives / (true_positives + false_positives + 1e-5)
recall = true_positives / (true_positives + false_negatives + 1e-5)

st.metric("Precision", f"{precision:.2f}")
st.metric("Recall", f"{recall:.2f}")

# Chart
st.subheader("Anomaly Visualization")
fig, ax = plt.subplots(figsize=(10, 6))
sns.scatterplot(
    x="file_size_MB", y="transfer_time_sec", 
    hue="detected_anomaly", style="label", palette="coolwarm", data=df, ax=ax
)
plt.title("Anomaly Detection: File Size vs Transfer Time")
plt.grid(True)
st.pyplot(fig)

st.subheader("Sample Logs")
st.dataframe(df.head(20))