
import streamlit as st
import numpy as np
import time

from simulate_data import generate_vibration_data
from model import AnomalyDetector

st.set_page_config(page_title="Real-Time Vibration Monitoring", layout="wide")
st.title("🔧 Real-Time Spindle Vibration Anomaly Detection")

# Generate full vibration data once at start
total_points = 1000
time_data, vibration_data = generate_vibration_data(n_points=total_points, anomaly=True)

# Initialize model and fit on first 500 (assume normal)
model = AnomalyDetector()
model.fit(vibration_data[:500])

# Placeholders
chart = st.empty()
anomaly_placeholder = st.empty()

# Real-time loop simulation
window_size = 50
detected_count = 0

for i in range(500, total_points):
    window_signal = vibration_data[i-window_size:i]
    current_time = time_data[i-window_size:i]
    
    anomaly_flags = model.predict(window_signal)
    is_anomaly = anomaly_flags[-1] == 1

    chart.line_chart({"Vibration": window_signal})

    if is_anomaly:
        detected_count += 1
        anomaly_placeholder.warning(f"⚠️ Anomaly Detected at t={i}", icon="⚠️")
    else:
        anomaly_placeholder.success(f"No Anomaly at t={i}")

    time.sleep(0.2)
    
st.success(f"✅ Real-Time Monitoring Complete. Total Anomalies: {detected_count}")
