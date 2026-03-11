import numpy as np
from tensorflow.keras.models import load_model
import joblib, os, math

MODEL = "models/improved/improved_best.h5"
SEQ = "artifacts/sequences_improved.npz"
SCALER = "artifacts/scaler_improved.pkl"

for f in [MODEL, SEQ, SCALER]:
    if not os.path.exists(f):
        print("Missing:", f); raise SystemExit

d = np.load(SEQ)
X = d["X"]; y = d["y"]
model = load_model(MODEL, compile=False)
scaler = joblib.load(SCALER)

y_pred = model.predict(X, verbose=0)
n_features = X.shape[2]
y_true = y.reshape(-1, n_features)
y_pred_r = y_pred.reshape(-1, n_features)

y_true_inv = scaler.inverse_transform(y_true)
y_pred_inv = scaler.inverse_transform(y_pred_r)

mse = np.mean((y_true_inv - y_pred_inv)**2)
rmse = math.sqrt(mse)
print("Improved model RMSE (degrees):", rmse)
