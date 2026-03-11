import os, math
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
import plotly.graph_objects as go
from dash import Dash, dcc, html, Output, Input

# Paths
PROCESSED_CSV = "data/processed/ibtracs_nio_cleaned.csv"
SCALER = "artifacts/scaler_improved.pkl"
MODEL = "models/improved/improved_best.h5"

FORECAST_STEPS_DEFAULT = 6
AUTO_REFRESH_MS = 8000     # 8 seconds
CONE_SIDES = 24            # number of points for uncertainty circle

# Load data
if not os.path.exists(PROCESSED_CSV):
    raise FileNotFoundError(PROCESSED_CSV)
df = pd.read_csv(PROCESSED_CSV)
df.columns = [c.upper().strip() for c in df.columns]

# Detect grouping column
for cand in ["SID","STORM_ID","NAME","TRACK_ID","ID"]:
    if cand in df.columns:
        GROUP_COL = cand
        break
else:
    df["_GRP"] = 1
    GROUP_COL = "_GRP"

# Features (same as model)
FEATURES = ["LAT", "LON"]

# Load scaler and model
scaler = joblib.load(SCALER)
model = load_model(MODEL, compile=False)

# Compute RMSE for uncertainty cone
def compute_rmse():
    npz = "artifacts/sequences_improved.npz"
    if not os.path.exists(npz):
        return 0.8
    d = np.load(npz)
    X = d["X"]; y = d["y"]
    y_pred = model.predict(X, verbose=0)
    n_features = X.shape[2]
    y_true = y.reshape(-1, n_features)
    y_pred_r = y_pred.reshape(-1, n_features)
    y_true_inv = scaler.inverse_transform(y_true)
    y_pred_inv = scaler.inverse_transform(y_pred_r)
    mse = np.mean((y_true_inv - y_pred_inv) ** 2)
    return math.sqrt(mse)

RMSE_DEG = compute_rmse()

# List available storms
def list_storms():
    if "ISO_TIME" in df.columns:
        temp = df.groupby(GROUP_COL)["ISO_TIME"].max().sort_values(ascending=False)
    else:
        temp = df.groupby(GROUP_COL).size().sort_values(ascending=False)
    return temp.index.tolist()[:200]

STORM_LIST = list_storms()

# Build last sequence
def get_last_sequence(sid, seq_len=12):
    g = df[df[GROUP_COL] == sid].copy()
    if g.empty:
        return None
    if "ISO_TIME" in g.columns:
        g = g.sort_values("ISO_TIME")
    arr = g[FEATURES].to_numpy(dtype=float)

    if len(arr) < seq_len:
        pad = np.repeat(arr[:1], seq_len - len(arr), axis=0)
        arr = np.vstack([pad, arr])
    else:
        arr = arr[-seq_len:]
    return arr

# Multi-step forecast
def rollout(seq_unscaled, steps=6):
    seq = seq_unscaled.copy()
    seq_len = seq.shape[0]
    preds = []

    for _ in range(steps):
        flat = seq.reshape(-1, len(FEATURES))
        scaled = scaler.transform(flat).reshape(1, seq_len, len(FEATURES))
        delta_scaled = model.predict(scaled, verbose=0)[0]
        delta = scaler.inverse_transform(delta_scaled.reshape(-1, len(FEATURES)))[0]
        last = seq[-1]
        next_pt = last + delta
        preds.append(next_pt.copy())
        seq = np.vstack([seq[1:], next_pt])
    return np.array(preds)

# Uncertainty cone around a lat/lon
def make_circle(lat, lon, radius_deg, sides=24):
    lats, lons = [], []
    for i in range(sides):
        ang = 2 * math.pi * i / sides
        lats.append(lat + radius_deg * math.cos(ang))
        lons.append(lon + radius_deg * math.sin(ang))
    lats.append(lats[0])
    lons.append(lons[0])
    return lats, lons

# DASH APP
app = Dash(__name__)

app.layout = html.Div([
    html.H2("Cyclone EWS — Multi-step Forecast Dashboard", style={"textAlign": "center"}),

    html.Div([
        html.Label("Select Storm (SID):"),
        dcc.Dropdown(
            id="storm-dropdown",
            options=[{"label": s, "value": s} for s in STORM_LIST],
            value=STORM_LIST[0],
            clearable=False
        )
    ], style={"width": "45%", "display": "inline-block"}),

    html.Div([
        html.Label("Forecast Steps:"),
        dcc.Slider(
            id="steps-slider",
            min=1, max=12, step=1,
            value=FORECAST_STEPS_DEFAULT,
            marks={i: str(i) for i in range(1, 13)}
        )
    ], style={"width": "45%", "display": "inline-block"}),

    dcc.Graph(id="map-graph"),
    dcc.Interval(id="refresh", interval=AUTO_REFRESH_MS, n_intervals=0)
])

@app.callback(
    Output("map-graph", "figure"),
    Input("storm-dropdown", "value"),
    Input("steps-slider", "value"),
    Input("refresh", "n_intervals")
)
def update_map(sid, steps, n):
    seq = get_last_sequence(sid, seq_len=12)
    if seq is None:
        return go.Figure()

    g = df[df[GROUP_COL] == sid].copy()
    if "ISO_TIME" in g.columns:
        g = g.sort_values("ISO_TIME")

    hist_lats = g["LAT"].tolist()
    hist_lons = g["LON"].tolist()

    preds = rollout(seq, steps=steps)
    pred_lats = preds[:, 0].tolist()
    pred_lons = preds[:, 1].tolist()

    fig = go.Figure()

    fig.add_trace(go.Scattermapbox(
        lat=hist_lats, lon=hist_lons,
        mode="lines+markers", name="History",
        line=dict(width=2, color="blue")
    ))

    fig.add_trace(go.Scattermapbox(
        lat=pred_lats, lon=pred_lons,
        mode="lines+markers", name="Forecast",
        line=dict(width=2, color="red")
    ))

    for plat, plon in zip(pred_lats, pred_lons):
        lts, lns = make_circle(plat, plon, RMSE_DEG, sides=CONE_SIDES)
        fig.add_trace(go.Scattermapbox(
            lat=lts, lon=lns, mode="lines", fill="toself",
            line=dict(width=1, color="rgba(255,0,0,0.5)"),
            fillcolor="rgba(255,0,0,0.1)", showlegend=False
        ))

    fig.update_layout(
        mapbox_style="open-street-map",
        mapbox_center={"lat": hist_lats[-1], "lon": hist_lons[-1]},
        mapbox_zoom=4,
        height=760
    )

    return fig

if __name__ == "__main__":
    app.run(debug=True)
