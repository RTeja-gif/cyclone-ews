import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from tensorflow.keras.models import load_model

# Load model & scaler
model = load_model("models/improved/improved_best.h5", compile=False)
scaler = joblib.load("artifacts/scaler_improved.pkl")

# Load sequences
d = np.load("artifacts/sequences_improved.npz")
X = d["X"]

# Use last sequence
x = X[-1:].copy()

n_features = X.shape[2]
steps = 12

# Inverse last known point
last_scaled = x[0, -1, :].reshape(1, -1)
last_unscaled = scaler.inverse_transform(last_scaled)[0]

predicted_points = []
current_seq = x.copy()

for _ in range(steps):
    pred_scaled = model.predict(current_seq, verbose=0)
    delta = scaler.inverse_transform(pred_scaled.reshape(1, -1))[0]
    next_point = last_unscaled + delta
    predicted_points.append(next_point)

    # update sequence
    next_scaled = scaler.transform(next_point.reshape(1, -1))[0]
    current_seq = np.roll(current_seq, -1, axis=1)
    current_seq[0, -1, :] = next_scaled
    last_unscaled = next_point

predicted_points = np.array(predicted_points)

# Plot
plt.figure(figsize=(6,6))
plt.plot(predicted_points[:,1], predicted_points[:,0],
         marker="o", color="red", label="Predicted Path")

plt.scatter(predicted_points[0,1], predicted_points[0,0],
            c="black", label="Forecast Start")

plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("Multi-Step Cyclone Trajectory Forecast (12 Steps)")
plt.legend()
plt.grid(alpha=0.3)

plt.savefig("paper/figures/fig7_multistep_forecast.png", dpi=300)
plt.close()
