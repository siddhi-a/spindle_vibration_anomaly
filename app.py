
import streamlit as st
import pandas as pd
import joblib

st.title("Spindle Vibration Anomaly Detection")

df = pd.read_csv("vibration_logs.csv")
model = joblib.load("model.pkl")
X = df[['Vibration_X', 'Vibration_Y', 'Vibration_Z']]
df['Anomaly'] = model.predict(X)
df['Anomaly'] = df['Anomaly'].map({1: 'Normal', -1: 'Anomaly'})

st.dataframe(df)
st.line_chart(df[['Vibration_X', 'Vibration_Y', 'Vibration_Z']])
