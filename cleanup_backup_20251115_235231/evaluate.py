import numpy as np
from tensorflow.keras.models import load_model
import joblib
import math
import os, sys

MODEL_PATH = "models/lstm_best.h5"
SEQ_PATH = "artifacts/sequences.npz"
SCALER_PATH = "artifacts/scaler.pkl"

# check files
for p in [MODEL_PATH, SEQ_PATH, SCALER_PATH]:
    if not os.path.exists(p):
        print("ERROR: missing:", p); sys.exit(1)

# load
d = np.load(SEQ_PATH)
X = d["X"]
y = d["y"]

model = load_model(MODEL_PATH, compile=False)
scaler = joblib.load(SCALER_PATH)

# predict
y_pred = model.predict(X, verbose=0)

# reshape for inverse scaling
n_features = X.shape[2]
y_true = y.reshape(-1, n_features)
y_pred_r = y_pred.reshape(-1, n_features)

# inverse scale
y_true_inv = scaler.inverse_transform(y_true)
y_pred_inv = scaler.inverse_transform(y_pred_r)

# compute RMSE manually
mse = np.mean((y_true_inv - y_pred_inv)**2)
rmse = math.sqrt(mse)

print("\n======================================")
print(" RMSE (degrees for lat/lon):", rmse)
print("======================================\n")
