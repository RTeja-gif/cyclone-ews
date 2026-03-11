import os
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import plotly.graph_objects as go
from dash import Dash, dcc, html

# paths
SEQ = "artifacts/sequences_improved.npz"
SCALER = "artifacts/scaler_improved.pkl"
MODEL = "models/improved/improved_best.h5"

# load sequences and model
d = np.load(SEQ)
X = d["X"]
model = load_model(MODEL, compile=False)
scaler = joblib.load(SCALER)

# take last sequence
x = X[-1:]

# model predicts delta (scaled)
pred_scaled = model.predict(x, verbose=0)
n_features = X.shape[2]
pred = pred_scaled.reshape(-1, n_features)
pred_inv = scaler.inverse_transform(pred)

# last known real position
last_scaled = x[0, -1, :].reshape(1, -1)
last_unscaled = scaler.inverse_transform(last_scaled)[0]
last_lat, last_lon = float(last_unscaled[0]), float(last_unscaled[1])

# predicted next position
next_lat = float(last_lat + pred_inv[0][0])
next_lon = float(last_lon + pred_inv[0][1])

app = Dash(__name__)

# Create figure using Scattermap (maplibre-compatible)
fig = go.Figure()

# Last known point (blue)
fig.add_trace(go.Scattermap(
    lat=[last_lat],
    lon=[last_lon],
    mode="markers",
    marker=dict(size=12, color="blue"),
    name="Last Known Position"
))

# Predicted next point (red)
fig.add_trace(go.Scattermap(
    lat=[next_lat],
    lon=[next_lon],
    mode="markers",
    marker=dict(size=14, color="red"),
    name="Predicted Next Position"
))

# Direction line (green)
fig.add_trace(go.Scattermap(
    lat=[last_lat, next_lat],
    lon=[last_lon, next_lon],
    mode="lines",
    line=dict(width=4, color="green"),
    name="Predicted Direction"
))

# map settings
fig.update_layout(
    map_style="open-street-map",
    map_zoom=4,
    map_center={"lat": last_lat, "lon": last_lon},
    height=700
)

app.layout = html.Div([
    html.H1("Cyclone Track & Forecast Dashboard", style={"textAlign": "center"}),
    dcc.Graph(id="cyclone-map", figure=fig)
])

if __name__ == "__main__":
    app.run(debug=True)
