
import pandas as pd
from sklearn.ensemble import IsolationForest
import joblib

df = pd.read_csv("vibration_logs.csv")
X = df[['Vibration_X', 'Vibration_Y', 'Vibration_Z']]
model = IsolationForest(contamination=0.1, random_state=42)
model.fit(X)
joblib.dump(model, "model.pkl")
print("Model trained and saved as model.pkl")
