import streamlit as st
import pandas as pd
import joblib
import os
from sklearn.ensemble import IsolationForest

st.set_page_config(page_title="Spindle Vibration Anomaly Detection", layout="centered")

st.title("🔍 Spindle Vibration Anomaly Detection")
st.markdown("Detect abnormal vibration patterns to prevent costly breakdowns in heavy milling machines.")

# Load data
df = pd.read_csv("vibration_logs.csv")

# Features
X = df[['Vibration_X', 'Vibration_Y', 'Vibration_Z']]

# Check if model.pkl exists, if not, train and save it
if not os.path.exists("model.pkl"):
    st.warning("Model not found. Training model now...")
    model = IsolationForest(contamination=0.1, random_state=42)
    model.fit(X)
    joblib.dump(model, "model.pkl")
    st.success("Model trained and saved!")
else:
    model = joblib.load("model.pkl")

# Predict anomalies
df['Anomaly'] = model.predict(X)
df['Anomaly'] = df['Anomaly'].map({1: '✅ Normal', -1: '🚨 Anomaly'})

# Display data
st.subheader("📊 Vibration Data with Anomaly Detection")
st.dataframe(df)

# Line Chart
st.subheader("📈 Vibration Trends")
st.line_chart(df[['Vibration_X', 'Vibration_Y', 'Vibration_Z']])

# Optional: Count of anomalies
anomaly_count = df['Anomaly'].value_counts().to_dict()
st.subheader("📌 Anomaly Summary")
st.markdown(f"- ✅ Normal Points: **{anomaly_count.get('✅ Normal', 0)}**")
st.markdown(f"- 🚨 Anomalies Detected: **{anomaly_count.get('🚨 Anomaly', 0)}**")
