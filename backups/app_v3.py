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

# UI defaults
FORECAST_DEFAULT = 6
REFRESH_MS = 8000
CONE_POINTS = 24
SIDEBAR_WIDTH = "320px"

# Load data & model
df = pd.read_csv(PROCESSED_CSV)
df.columns = [c.upper().strip() for c in df.columns]
scaler = joblib.load(SCALER)
model = load_model(MODEL, compile=False)

# Group column detection
for cand in ["SID","STORM_ID","NAME","TRACK_ID","ID"]:
    if cand in df.columns:
        GROUP_COL = cand
        break
else:
    df["_GRP"] = 1
    GROUP_COL = "_GRP"

FEATURES = ["LAT","LON"]

def compute_rmse(npz="artifacts/sequences_improved.npz"):
    if not os.path.exists(npz): return 0.9
    d = np.load(npz); X=d["X"]; y=d["y"]
    y_pred = model.predict(X, verbose=0)
    n_features = X.shape[2]
    y_true = y.reshape(-1,n_features); y_pred_r = y_pred.reshape(-1,n_features)
    y_true_inv = scaler.inverse_transform(y_true); y_pred_inv = scaler.inverse_transform(y_pred_r)
    return math.sqrt(np.mean((y_true_inv - y_pred_inv)**2))

RMSE_DEG = compute_rmse()

def list_storms(limit=200):
    if "ISO_TIME" in df.columns:
        temp = df.groupby(GROUP_COL)["ISO_TIME"].max().sort_values(ascending=False)
    else:
        temp = df.groupby(GROUP_COL).size().sort_values(ascending=False)
    return temp.index.tolist()[:limit]

STORMS = list_storms()

def get_last_sequence(sid, seq_len=12):
    g = df[df[GROUP_COL]==sid].copy()
    if g.empty: return None
    if "ISO_TIME" in g.columns: g = g.sort_values("ISO_TIME")
    arr = g[FEATURES].to_numpy(dtype=float)
    if len(arr) < seq_len:
        pad = np.repeat(arr[:1], seq_len-len(arr), axis=0)
        arr = np.vstack([pad, arr])
    else:
        arr = arr[-seq_len:]
    return arr

def rollout(seq, steps=6):
    seq_local = seq.copy()
    seq_len = seq_local.shape[0]
    preds = []
    for _ in range(steps):
        flat = seq_local.reshape(-1, len(FEATURES))
        scaled = scaler.transform(flat).reshape(1, seq_len, len(FEATURES))
        delta_scaled = model.predict(scaled, verbose=0)[0]
        delta = scaler.inverse_transform(delta_scaled.reshape(-1,len(FEATURES)))[0]
        last = seq_local[-1]; nxt = last + delta
        preds.append(nxt.copy())
        seq_local = np.vstack([seq_local[1:], nxt])
    return np.array(preds)

def make_circle(lat, lon, radius_deg, sides=CONE_POINTS):
    lats=[]; lons=[]
    for i in range(sides):
        ang = 2*math.pi*i/sides
        lats.append(lat + radius_deg*math.cos(ang))
        lons.append(lon + radius_deg*math.sin(ang))
    lats.append(lats[0]); lons.append(lons[0])
    return lats, lons

# App layout with fixed left sidebar
app = Dash(__name__)
app.layout = html.Div(style={"display":"flex","height":"100vh","fontFamily":"Arial"}, children=[
    html.Div(id="sidebar", style={"width":SIDEBAR_WIDTH,"padding":"18px","boxShadow":"2px 0 6px rgba(0,0,0,0.1)","backgroundColor":"#fafafa","overflow":"auto"}, children=[
        html.H3("Control Panel", style={"marginTop":"0"}),
        html.Label("Storm (SID)"),
        dcc.Dropdown(id="sid-dropdown", options=[{"label":s,"value":s} for s in STORMS], value=STORMS[0] if len(STORMS)>0 else None, clearable=False, style={"marginBottom":"12px"}),
        html.Label("Forecast Steps"),
        dcc.Slider(id="steps", min=1, max=12, step=1, value=FORECAST_DEFAULT, marks={i:str(i) for i in range(1,13)}, tooltip={"placement":"bottom","always_visible":False}),
        html.Hr(),
        html.Div([
            html.Div([html.Strong("RMSE (deg): "), html.Span(id="rmse-text", children=f"{RMSE_DEG:.3f}")]),
            html.Br(),
            html.Div([html.Strong("RMSE (km est): "), html.Span(id="rmse-km", children=f"{RMSE_DEG*111:.0f} km")]),
        ], style={"marginBottom":"12px"}),
        html.Div(id="last-info", style={"marginTop":"8px","lineHeight":"1.6"}),
        html.Hr(),
        html.Div(id="alerts-box", children=[
            html.H4("Alerts", style={"margin":"6px 0 6px 0"}),
            html.Div("No active alerts", id="alerts-text", style={"color":"#2b7a78"})
        ]),
        html.Div(style={"height":"24px"})
    ]),
    html.Div(style={"flex":"1","padding":"8px"}, children=[
        html.H2("Cyclone EWS — Map", style={"textAlign":"center", "marginTop":"6px"}),
        dcc.Graph(id="map", style={"height":"calc(100vh - 120px)"}),
        dcc.Interval(id="refresh-interval", interval=REFRESH_MS, n_intervals=0)
    ])
])

@app.callback(
    Output("map", "figure"),
    Output("last-info", "children"),
    Output("alerts-text", "children"),
    Output("rmse-text", "children"),
    Output("rmse-km", "children"),
    Input("sid-dropdown","value"),
    Input("steps","value"),
    Input("refresh-interval","n_intervals")
)
def update_ui(sid, steps, n):
    if sid is None:
        return go.Figure(), "No storm selected", "No data", f"{RMSE_DEG:.3f}", f"{RMSE_DEG*111:.0f} km"
    seq = get_last_sequence(sid, seq_len=12)
    if seq is None:
        return go.Figure(), "No data for selected storm", "No data", f"{RMSE_DEG:.3f}", f"{RMSE_DEG*111:.0f} km"
    # historical
    g = df[df[GROUP_COL]==sid].copy()
    if "ISO_TIME" in g.columns:
        g = g.sort_values("ISO_TIME")
    hist_lats = g["LAT"].tolist(); hist_lons = g["LON"].tolist()
    last_time = str(g["ISO_TIME"].max()) if "ISO_TIME" in g.columns else "N/A"
    last_info = html.Div([html.Div(["SID: ", html.B(sid)]), html.Div(["Last obs: ", html.B(last_time)]), html.Div(["Points: ", len(g)])])
    # forecasts
    preds = rollout(seq, steps=steps)
    pred_lats = preds[:,0].tolist(); pred_lons = preds[:,1].tolist()
    fig = go.Figure()
    fig.add_trace(go.Scattermapbox(lat=hist_lats, lon=hist_lons, mode="lines+markers", name="History", line=dict(width=2,color="blue")))
    fig.add_trace(go.Scattermapbox(lat=[hist_lats[-1]], lon=[hist_lons[-1]], mode="markers", marker=dict(size=12,color="darkblue"), name="Last Known"))
    if len(pred_lats)>0:
        fig.add_trace(go.Scattermapbox(lat=pred_lats, lon=pred_lons, mode="lines+markers", name="Forecast", line=dict(width=2,color="red")))
    for plat, plon in zip(pred_lats, pred_lons):
        lts, lns = make_circle(plat, plon, RMSE_DEG, sides=CONE_POINTS)
        fig.add_trace(go.Scattermapbox(lat=lts, lon=lns, mode="lines", fill="toself", line=dict(width=1,color="rgba(255,0,0,0.5)"), fillcolor="rgba(255,0,0,0.1)", showlegend=False))
    center_lat = hist_lats[-1] if len(hist_lats)>0 else 15.0
    center_lon = hist_lons[-1] if len(hist_lons)>0 else 85.0
    fig.update_layout(mapbox_style="open-street-map", mapbox_center={"lat":center_lat,"lon":center_lon}, mapbox_zoom=4, margin={"r":0,"t":0,"l":0,"b":0})
    # simple alert logic: if forecast moves >0.6 deg (~66km) mark "Potential significant movement"
    alert_msg = "No active alerts"
    if len(pred_lats)>0:
        last = np.array([hist_lats[-1], hist_lons[-1]])
        dist = np.mean(np.sqrt((preds[:,0]-last[0])**2 + (preds[:,1]-last[1])**2))
        if dist > 0.6:
            alert_msg = "⚠️ Significant forecast movement detected"
    return fig, last_info, alert_msg, f"{RMSE_DEG:.3f}", f"{RMSE_DEG*111:.0f} km"

if __name__ == "__main__":
    app.run(debug=True)
