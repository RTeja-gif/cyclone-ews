import numpy as np, joblib, os, sys
from tensorflow.keras.models import load_model

SEQ_FILE = "artifacts/sequences_improved.npz"
SCALER_FILE = "artifacts/scaler_improved.pkl"
MODEL_FILE = "models/improved/improved_best.h5"

for f in [SEQ_FILE, SCALER_FILE, MODEL_FILE]:
    if not os.path.exists(f):
        print("Missing:", f); sys.exit(1)

d = np.load(SEQ_FILE); X = d["X"]
model = load_model(MODEL_FILE, compile=False)
scaler = joblib.load(SCALER_FILE)

x = X[-1:]
pred_scaled = model.predict(x, verbose=0)
n_features = X.shape[2]
pred = pred_scaled.reshape(-1, n_features)
pred_inv = scaler.inverse_transform(pred)
last_scaled = x[0,-1,:].reshape(1,-1)
last_unscaled = scaler.inverse_transform(last_scaled)[0]
next_pos = last_unscaled + pred_inv[0]

print("Last:", last_unscaled[:2])
print("Predicted next:", next_pos[:2])
