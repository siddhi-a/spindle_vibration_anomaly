import streamlit as st
import pandas as pd
import joblib
import os
import time
from sklearn.ensemble import IsolationForest

st.set_page_config(page_title="Spindle Vibration Anomaly Detection", layout="wide")

st.title("🛠️ Spindle Vibration Anomaly Detection")
st.markdown("Early detection of abnormal vibration patterns in heavy milling machines using AI.")

# Sidebar for CSV upload
st.sidebar.header("📂 Upload Vibration CSV")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")
st.sidebar.markdown("⚠️ CSV must contain: `Vibration_X`, `Vibration_Y`, `Vibration_Z`")

# Load default or uploaded data
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

# Predict full dataset for summary
df['Anomaly'] = model.predict(X)
df['Anomaly'] = df['Anomaly'].map({1: '✅ Normal', -1: '🚨 Anomaly'})

# Summary
st.subheader("📋 Anomaly Summary")
col1, col2 = st.columns(2)
with col1:
    st.metric("✅ Normal", df['Anomaly'].value_counts().get('✅ Normal', 0))
with col2:
    st.metric("🚨 Anomaly", df['Anomaly'].value_counts().get('🚨 Anomaly', 0))

# Full data table
st.subheader("📈 Full Vibration Dataset with Anomaly Detection")
st.dataframe(df, use_container_width=True)

# Optional Real-Time Simulation (Triggered)
st.subheader("⏱️ Real-Time Vibration Monitoring (Simulated)")
if st.button("▶️ Start Simulation"):
    simulation_df = df.copy()
    placeholder = st.empty()
    for i in range(min(len(simulation_df), 50)):  # limit to 50 rows for cloud safety
        sim_data = simulation_df.iloc[:i+1]
        placeholder.line_chart(sim_data[['Vibration_X', 'Vibration_Y', 'Vibration_Z']])
        placeholder.dataframe(sim_data.tail(5), use_container_width=True)
        time.sleep(0.4)
    st.success("✅ Real-time simulation finished.")
else:
    st.info("Press the ▶️ Start Simulation button to begin real-time visualization.")

