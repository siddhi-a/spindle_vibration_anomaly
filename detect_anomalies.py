
import pandas as pd
import joblib

df = pd.read_csv("vibration_logs.csv")
X = df[['Vibration_X', 'Vibration_Y', 'Vibration_Z']]
model = joblib.load("model.pkl")
df['anomaly'] = model.predict(X)
df['anomaly'] = df['anomaly'].map({1: 'Normal', -1: 'Anomaly'})
print(df[['Timestamp', 'Vibration_X', 'Vibration_Y', 'Vibration_Z', 'anomaly']])
