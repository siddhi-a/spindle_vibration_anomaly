
import streamlit as st
import numpy as np

from simulate_data import generate_vibration_data
from model import AnomalyDetector

st.title("Spindle Vibration Anomaly Detection")
st.markdown("Detect real-time anomalies in spindle vibration signals.")

time_steps, vibration = generate_vibration_data(n_points=3000, anomaly=True)

model = AnomalyDetector()
model.fit(vibration)
anomalies = model.predict(vibration)

st.line_chart(data={"Vibration": vibration})

num_anomalies = np.sum(anomalies)
st.metric("Detected Anomalies", num_anomalies)
