
import numpy as np
from sklearn.ensemble import IsolationForest

class AnomalyDetector:
    def __init__(self):
        self.model = IsolationForest(contamination=0.05)

    def fit(self, signal):
        X = signal.reshape(-1, 1)
        self.model.fit(X)

    def predict(self, signal):
        X = signal.reshape(-1, 1)
        preds = self.model.predict(X)
        return np.where(preds == -1, 1, 0)
