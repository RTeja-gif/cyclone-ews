import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import train_test_split

DATA = "artifacts/sequences_improved.npz"
OUT_DIR = "models/improved"
os.makedirs(OUT_DIR, exist_ok=True)

d = np.load(DATA)
X = d["X"]
y = d["y"]

# split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

seq_len = X.shape[1]
n_features = X.shape[2]
output_dim = y.shape[1]

model = Sequential([
    LSTM(128, input_shape=(seq_len, n_features), return_sequences=False),
    Dropout(0.2),
    Dense(64, activation="relu"),
    Dropout(0.1),
    Dense(output_dim)
])

model.compile(optimizer="adam", loss="mse", metrics=["mse"])
print("Training improved model: epochs=15, batch_size=16")

chk = ModelCheckpoint(os.path.join(OUT_DIR, "improved_best.h5"), save_best_only=True, monitor="val_loss")
es = EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True)

history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                    epochs=15, batch_size=16, callbacks=[chk, es])

model.save(os.path.join(OUT_DIR, "improved_final.h5"))
print("Saved improved model to:", OUT_DIR)
