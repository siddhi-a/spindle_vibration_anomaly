
import streamlit as st
import pandas as pd
import joblib
import os
import time
from sklearn.ensemble import IsolationForest

st.set_page_config(page_title="Spindle Vibration Anomaly Detection", layout="wide")

st.title("🛠️ Spindle Vibration Anomaly Detection")
st.markdown("Early detection of abnormal vibration patterns in heavy milling machines using AI.")

# Sidebar
st.sidebar.header("📂 Upload Vibration CSV")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")
st.sidebar.markdown("⚠️ CSV must contain: `Vibration_X`, `Vibration_Y`, `Vibration_Z`")

# Load data: uploaded file OR default CSV OR error out
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("✅ Custom data uploaded successfully!")
elif os.path.exists("vibration_logs.csv"):
    df = pd.read_csv("vibration_logs.csv")
    st.info("ℹ️ Using default sample data (vibration_logs.csv)")
else:
    st.error("❌ No vibration data found. Please upload a CSV file to continue.")
    st.stop()

# Features
X = df[['Vibration_X', 'Vibration_Y', 'Vibration_Z']]

# Train or load model
if not os.path.exists("model.pkl"):
    st.warning("Training model (no saved model found)...")
    model = IsolationForest(contamination=0.1, random_state=42)
    model.fit(X)
    joblib.dump(model, "model.pkl")
    st.success("Model trained and saved as model.pkl!")
else:
    model = joblib.load("model.pkl")

# Real-time simulation section
st.subheader("⏱️ Real-Time Vibration Monitoring (Simulated)")
placeholder = st.empty()
with placeholder.container():
    for i in range(len(df)):
        single_row = df.iloc[:i+1].copy()
        single_row['Anomaly'] = model.predict(single_row[['Vibration_X', 'Vibration_Y', 'Vibration_Z']])
        single_row['Anomaly'] = single_row['Anomaly'].map({1: '✅ Normal', -1: '🚨 Anomaly'})

        st.line_chart(single_row[['Vibration_X', 'Vibration_Y', 'Vibration_Z']])
        st.dataframe(single_row.tail(5), use_container_width=True)
        time.sleep(0.3)
        if i == len(df) - 1:
            st.success("🔍 Real-time simulation complete.")
        placeholder.empty()

# Anomaly summary
df['Anomaly'] = model.predict(X)
df['Anomaly'] = df['Anomaly'].map({1: '✅ Normal', -1: '🚨 Anomaly'})

st.subheader("📋 Anomaly Summary")
col1, col2 = st.columns(2)
with col1:
    st.metric("✅ Normal", df['Anomaly'].value_counts().get('✅ Normal', 0))
with col2:
    st.metric("🚨 Anomaly", df['Anomaly'].value_counts().get('🚨 Anomaly', 0))

# Final data table
st.subheader("📈 Full Vibration Dataset with Anomaly Detection")
st.dataframe(df, use_container_width=True)
