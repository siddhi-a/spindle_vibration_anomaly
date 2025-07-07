
import streamlit as st
import numpy as np
import pandas as pd
import time

from model import AnomalyDetector

st.set_page_config(page_title="Real-Time Vibration Monitoring", layout="wide")
st.title("📈 Real-Time Spindle Vibration Anomaly Detection with CSV Upload")

uploaded_file = st.file_uploader("📂 Upload Vibration CSV File", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    if 'Vibration' not in df.columns:
        st.error("CSV must contain a 'Vibration' column.")
    else:
        vibration_data = df['Vibration'].values
        total_points = len(vibration_data)
        time_data = df[df.columns[0]].values

        # Train model on first 500 (assume normal)
        model = AnomalyDetector()
        model.fit(vibration_data[:500])

        chart = st.empty()
        anomaly_placeholder = st.empty()

        window_size = 50
        detected_count = 0

        for i in range(500, total_points):
            window_signal = vibration_data[i-window_size:i]
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
else:
    st.info("Please upload a CSV file with vibration data to begin.")
