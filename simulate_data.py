
import numpy as np

def generate_vibration_data(n_points=1000, anomaly=False):
    time = np.arange(n_points)
    vibration = np.sin(0.02 * time) + np.random.normal(0, 0.1, n_points)
    
    if anomaly:
        start = np.random.randint(n_points // 2, n_points - 100)
        vibration[start:start + 50] += np.random.normal(3, 0.5, 50)
    
    return time, vibration
