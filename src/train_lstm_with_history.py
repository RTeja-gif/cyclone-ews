import os
import json
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split

# Load sequences
d = np.load("artifacts/sequences_improved.npz")
X = d["X"]
y = d["y"]

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

seq_len = X.shape[1]
n_features = X.shape[2]
out_dim = y.shape[1]

model = Sequential([
    LSTM(64, input_shape=(seq_len, n_features)),
    Dropout(0.2),
    Dense(out_dim)
])

model.compile(optimizer="adam", loss="mse")

print("Training LSTM for 5 epochs (CPU-friendly)...")

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=5,
    batch_size=16,
    verbose=1
)

# Save history
os.makedirs("artifacts", exist_ok=True)
with open("artifacts/training_history.json", "w") as f:
    json.dump(history.history, f)

model.save("models/lstm_for_paper.h5")

print("Training complete. History saved.")
