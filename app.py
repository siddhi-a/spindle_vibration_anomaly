import streamlit as st
import pandas as pd
import joblib
import os
from sklearn.ensemble import IsolationForest

st.set_page_config(page_title="Spindle Vibration Anomaly Detection", layout="wide")

st.title("🛠️ Spindle Vibration Anomaly Detection")
st.markdown("Early detection of abnormal vibration patterns in heavy milling machines using AI.")

# Sidebar for CSV upload
st.sidebar.header("📂 Upload Vibration CSV")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")
st.sidebar.markdown("⚠️ CSV must contain: `Vibration_X`, `Vibration_Y`, `Vibration_Z`")

# Load data
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("✅ Custom data uploaded successfully!")
elif os.path.exists("vibration_logs.csv"):
    df = pd.read_csv("vibration_logs.csv")
    st.info("ℹ️ Using default sample data (vibration_logs.csv)")
else:
    st.error("❌ No data found. Please upload a CSV file.")
    st.stop()

# Prepare features
X = df[['Vibration_X', 'Vibration_Y', 'Vibration_Z']]

# Train or load model
if not os.path.exists("model.pkl"):
    model = IsolationForest(contamination=0.1, random_state=42)
    model.fit(X)
    joblib.dump(model, "model.pkl")
else:
    model = joblib.load("model.pkl")

# Predict anomalies
df['Anomaly'] = model.predict(X)
df['Anomaly'] = df['Anomaly'].map({1: '✅ Normal', -1: '🚨 Anomaly'})

# Summary metrics
st.subheader("📋 Anomaly Summary")
col1, col2 = st.columns(2)
col1.metric("✅ Normal", df['Anomaly'].value_counts().get('✅ Normal', 0))
col2.metric("🚨 Anomaly", df['Anomaly'].value_counts().get('🚨 Anomaly', 0))

# Charts
st.subheader("📈 Vibration Trends")
st.line_chart(df[['Vibration_X', 'Vibration_Y', 'Vibration_Z']])

# Final table
st.subheader("📄 Full Dataset with Anomaly Labels")
st.dataframe(df, use_container_width=True)


