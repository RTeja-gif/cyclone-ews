import numpy as np
import joblib
from tensorflow.keras.models import load_model
import os, sys

SEQ_FILE = "artifacts/sequences.npz"
SCALER_FILE = "artifacts/scaler.pkl"
MODEL_FILE = "models/lstm_best.h5"

# check files
for f in [SEQ_FILE, SCALER_FILE, MODEL_FILE]:
    if not os.path.exists(f):
        print("Missing:", f)
        sys.exit(1)

# load data
d = np.load(SEQ_FILE)
X = d["X"]
model = load_model(MODEL_FILE, compile=False)
scaler = joblib.load(SCALER_FILE)

# use last sequence sample
x = X[-1:]

# model prediction (scaled deltas)
pred_scaled = model.predict(x, verbose=0)
n_features = X.shape[2]

# reshape for inverse scaling
pred = pred_scaled.reshape(-1, n_features)
pred_inv = scaler.inverse_transform(pred)

# last known position (unscaled)
last_scaled = x[0, -1, :].reshape(1, -1)
last_unscaled = scaler.inverse_transform(last_scaled)[0]

# compute next absolute lat/lon
next_position = last_unscaled + pred_inv[0]

print("\n================ CYCLONE FORECAST ================")
print("Last Known Position (lat, lon):", last_unscaled[:2])
print("Predicted Next Position (lat, lon):", next_position[:2])
print("==================================================\n")
