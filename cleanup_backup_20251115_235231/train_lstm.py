import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split

DATA = "artifacts/sequences.npz"
OUT_DIR = "models"
os.makedirs(OUT_DIR, exist_ok=True)

# Load sequences
d = np.load(DATA)
X = d["X"]      # shape: (N, seq_len, features)
y = d["y"]      # shape: (N, features)

# Train/validation split
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True
)

seq_len = X.shape[1]
n_features = X.shape[2]
output_dim = y.shape[1]

# Build model
model = Sequential([
    LSTM(64, input_shape=(seq_len, n_features)),
    Dense(32, activation="relu"),
    Dense(output_dim)
])

model.compile(optimizer="adam", loss="mse", metrics=["mse"])

print("Training model with epochs=5, batch_size=16...")

# Save best model on validation loss
checkpoint = ModelCheckpoint(
    os.path.join(OUT_DIR, "lstm_best.h5"),
    save_best_only=True,
    monitor="val_loss"
)

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=5,
    batch_size=16,
    callbacks=[checkpoint]
)

model.save(os.path.join(OUT_DIR, "lstm_final.h5"))
print("Model training complete. Saved to:", OUT_DIR)
