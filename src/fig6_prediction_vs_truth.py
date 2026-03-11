import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from tensorflow.keras.models import load_model

# Load data
df = pd.read_csv("data/processed/ibtracs_nio_cleaned.csv")
df.columns = [c.upper() for c in df.columns]

# Load model and scaler
model = load_model("models/improved/improved_best.h5", compile=False)
scaler = joblib.load("artifacts/scaler_improved.pkl")

# Load sequences
d = np.load("artifacts/sequences_improved.npz")
X = d["X"]
y = d["y"]

# Take last sample
x = X[-1:]
y_true = y[-1:]

# Predict
y_pred = model.predict(x, verbose=0)

# Inverse scaling
n_features = X.shape[2]
last_scaled = x[0, -1, :].reshape(1, -1)
last_unscaled = scaler.inverse_transform(last_scaled)[0]

true_delta = scaler.inverse_transform(y_true.reshape(1, -1))[0]
pred_delta = scaler.inverse_transform(y_pred.reshape(1, -1))[0]

true_next = last_unscaled + true_delta
pred_next = last_unscaled + pred_delta

# Plot
plt.figure(figsize=(6,6))
plt.scatter(last_unscaled[1], last_unscaled[0], c="black", label="Last Known Position")
plt.scatter(true_next[1], true_next[0], c="green", label="Actual Next Position")
plt.scatter(pred_next[1], pred_next[0], c="red", label="Predicted Next Position")

plt.plot(
    [last_unscaled[1], true_next[1]],
    [last_unscaled[0], true_next[0]],
    color="green"
)
plt.plot(
    [last_unscaled[1], pred_next[1]],
    [last_unscaled[0], pred_next[0]],
    color="red"
)

plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("Single-Step Cyclone Forecast: Prediction vs Ground Truth")
plt.legend()
plt.grid(alpha=0.3)

plt.savefig("paper/figures/fig6_prediction_vs_truth.png", dpi=300)
plt.close()
