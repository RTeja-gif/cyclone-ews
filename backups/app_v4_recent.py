import os, math
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
import plotly.graph_objects as go
from dash import Dash, dcc, html, Output, Input
from src.wind_hazard import estimate_wind_hazard

# Paths
PROCESSED_CSV = "data/processed/ibtracs_nio_cleaned.csv"
SCALER = "artifacts/scaler_improved.pkl"
MODEL = "models/improved/improved_best.h5"
SEQ_NPZ = "artifacts/sequences_improved.npz"

# UI defaults
FORECAST_DEFAULT = 6
REFRESH_MS = 8000
CONE_POINTS = 24
SIDEBAR_WIDTH = "340px"

# Load data & model
df = pd.read_csv(PROCESSED_CSV)
df.columns = [c.upper().strip() for c in df.columns]

# ensure SEASON exists and is int when possible
if "SEASON" in df.columns:
    df["SEASON_INT"] = pd.to_numeric(df["SEASON"], errors="coerce").astype("Int64")
else:
    df["SEASON_INT"] = pd.NA

scaler = joblib.load(SCALER)
model = load_model(MODEL, compile=False)

# detect group col
for cand in ["SID","STORM_ID","NAME","TRACK_ID","ID"]:
    if cand in df.columns:
        GROUP_COL = cand
        break
else:
    df["_GRP"]=1; GROUP_COL="_GRP"

FEATURES = ["LAT","LON"]

# compute RMSE for cones
def compute_rmse(npzpath=SEQ_NPZ):
    if not os.path.exists(npzpath):
        return 0.9
    d=np.load(npzpath); X=d["X"]; y=d["y"]
    y_pred = model.predict(X, verbose=0)
    n_features = X.shape[2]
    y_true = y.reshape(-1, n_features)
    y_pred_r = y_pred.reshape(-1, n_features)
    y_true_inv = scaler.inverse_transform(y_true)
    y_pred_inv = scaler.inverse_transform(y_pred_r)
    mse = np.mean((y_true_inv - y_pred_inv)**2)
    return math.sqrt(mse)

RMSE_DEG = compute_rmse()

# helpers
def list_storms_recent(limit=200, year_min=2000, year_max=2025):
    # select storms whose season or latest ISO_TIME year falls into the range
    candidates = []
    grouped = df.groupby(GROUP_COL)
    for sid, g in grouped:
        year_ok = False
        # check SEASON_INT if present
        if "SEASON_INT" in g.columns and pd.notna(g["SEASON_INT"].iloc[0]):
            try:
                y = int(g["SEASON_INT"].iloc[0])
                if year_min <= y <= year_max:
                    year_ok = True
            except:
                year_ok = False
        # fallback: check latest ISO_TIME year
        if not year_ok and "ISO_TIME" in g.columns:
            try:
                g_times = pd.to_datetime(g["ISO_TIME"], errors="coerce")
                if g_times.notna().any():
                    y = int(g_times.max().year)
                    if year_min <= y <= year_max:
                        year_ok = True
            except:
                year_ok = False
        if year_ok:
            candidates.append((sid, g.shape[0], int(g["ISO_TIME"].max().year) if "ISO_TIME" in g.columns and pd.to_datetime(g["ISO_TIME"], errors="coerce").notna().any() else None))
    # sort by most recent year (if available) then by group size
    candidates_sorted = sorted(candidates, key=lambda x: ((x[2] if x[2] is not None else 0), x[1]), reverse=True)
    storms = [c[0] for c in candidates_sorted][:limit]
    return storms

STORMS = list_storms_recent(limit=200, year_min=2000, year_max=2025)
if len(STORMS) == 0:
    STORMS = df.groupby(GROUP_COL).size().sort_values(ascending=False).index.tolist()[:200]

def get_last_sequence(sid, seq_len=12):
    g = df[df[GROUP_COL]==sid].copy()
    if g.empty: return None
    if "ISO_TIME" in g.columns: g = g.sort_values("ISO_TIME")
    arr = g[FEATURES].to_numpy(dtype=float)
    if len(arr) < seq_len:
        pad = np.repeat(arr[:1], seq_len - len(arr), axis=0)
        arr = np.vstack([pad, arr])
    else:
        arr = arr[-seq_len:]
    return arr

def rollout(seq_unscaled, steps=6):
    seq = seq_unscaled.copy()
    seq_len = seq.shape[0]
    preds = []
    for _ in range(steps):
        flat = seq.reshape(-1, len(FEATURES))
        scaled = scaler.transform(flat).reshape(1, seq_len, len(FEATURES))
        delta_scaled = model.predict(scaled, verbose=0)[0]
        delta = scaler.inverse_transform(delta_scaled.reshape(-1, len(FEATURES)))[0]
        last = seq[-1]; next_pt = last + delta
        preds.append(next_pt.copy())
        seq = np.vstack([seq[1:], next_pt])
    return np.array(preds)

def make_circle(lat, lon, radius_deg, sides=CONE_POINTS):
    lats=[]; lons=[]
    for i in range(sides):
        ang = 2*math.pi*i/sides
        lats.append(lat + radius_deg*math.cos(ang))
        lons.append(lon + radius_deg*math.sin(ang))
    lats.append(lats[0]); lons.append(lons[0])
    return lats, lons

def compute_translation_speed_kmh(sid):
    g = df[df[GROUP_COL]==sid].copy()
    if len(g) < 2: return 0.0
    if "ISO_TIME" in g.columns:
        g = g.sort_values("ISO_TIME")
    a = g.iloc[-2]; b = g.iloc[-1]
    try:
        lat1, lon1 = float(a["LAT"]), float(a["LON"])
        lat2, lon2 = float(b["LAT"]), float(b["LON"])
    except:
        return 0.0
    avg_lat = np.deg2rad((lat1+lat2)/2.0)
    dx = (lon2 - lon1) * 111.320 * math.cos(avg_lat)
    dy = (lat2 - lat1) * 110.574
    dist_km = math.sqrt(dx*dx + dy*dy)
    if "ISO_TIME" in g.columns and pd.notna(a["ISO_TIME"]) and pd.notna(b["ISO_TIME"]):
        dt_h = (b["ISO_TIME"] - a["ISO_TIME"]).total_seconds() / 3600.0
        if dt_h > 0:
            return dist_km / dt_h
    return dist_km

# APP (same layout as v4)
app = Dash(__name__)
app.layout = html.Div(style={"display":"flex","height":"100vh","fontFamily":"Arial"}, children=[
    html.Div(style={"width":SIDEBAR_WIDTH,"padding":"16px","boxShadow":"2px 0 6px rgba(0,0,0,0.1)","backgroundColor":"#fbfbfb","overflow":"auto"}, children=[
        html.H3("Control Panel", style={"marginTop":"0"}),
        html.Label("Storm (SID)"),
        dcc.Dropdown(id="sid_dd", options=[{"label":s,"value":s} for s in STORMS], value=STORMS[0] if len(STORMS)>0 else None, clearable=False, style={"marginBottom":"12px"}),
        html.Label("Forecast Steps"),
        dcc.Slider(id="steps", min=1, max=12, step=1, value=FORECAST_DEFAULT, marks={i:str(i) for i in range(1,13)}),
        html.Hr(),
        html.Div([html.Strong("RMSE (deg): "), html.Span(id="rmse_text", children=f"{RMSE_DEG:.3f}")]),
        html.Br(),
        html.Div([html.Strong("RMSE (km est): "), html.Span(id="rmse_km", children=f"{RMSE_DEG*111:.0f} km")]),
        html.Hr(),
        html.Div([html.Strong("Wind Hazard"), html.Div(id="wind-hazard-box", style={"padding":"10px","borderRadius":"6px","marginTop":"8px"})]),
        html.Hr(),
        html.Div(id="last-info", style={"lineHeight":"1.6"})
    ]),
    html.Div(style={"flex":"1","padding":"6px"}, children=[
        html.H2("Cyclone EWS — Recent Storms (2000-2025)", style={"textAlign":"center","marginTop":"6px"}),
        dcc.Graph(id="map", style={"height":"calc(100vh - 120px)"}),
        dcc.Interval(id="refresh", interval=REFRESH_MS, n_intervals=0)
    ])
])

@app.callback(
    Output("map","figure"),
    Output("last-info","children"),
    Output("wind-hazard-box","children"),
    Output("rmse_text","children"),
    Output("rmse_km","children"),
    Input("sid_dd","value"),
    Input("steps","value"),
    Input("refresh","n_intervals")
)
def update_ui(sid, steps, n):
    if sid is None:
        return go.Figure(), "No storm selected", "No data", f"{RMSE_DEG:.3f}", f"{RMSE_DEG*111:.0f} km"
    seq = get_last_sequence(sid, seq_len=12)
    if seq is None:
        return go.Figure(), "No data for selected storm", "No data", f"{RMSE_DEG:.3f}", f"{RMSE_DEG*111:.0f} km"
    # history
    g = df[df[GROUP_COL]==sid].copy()
    if "ISO_TIME" in g.columns: g = g.sort_values("ISO_TIME")
    hist_lats = g["LAT"].tolist(); hist_lons = g["LON"].tolist()
    last_time = str(g["ISO_TIME"].max()) if "ISO_TIME" in g.columns else "N/A"
    last_info = html.Div([html.Div(["SID: ", html.B(sid)]), html.Div(["Last obs: ", html.B(last_time)]), html.Div(["Points: ", len(g)])])
    # forecast
    preds = rollout(seq, steps=steps)
    pred_lats = preds[:,0].tolist() if len(preds)>0 else []
    pred_lons = preds[:,1].tolist() if len(preds)>0 else []
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
    # wind hazard estimation
    translation_kmh = compute_translation_speed_kmh(sid)
    if len(preds)>0:
        last = seq[-1]
        deltas = preds - last
        mean_delta_deg = float(np.mean(np.linalg.norm(deltas[:,:2], axis=1)))
    else:
        mean_delta_deg = 0.0
    hazard_level, est_wind_kmh, color = estimate_wind_hazard(translation_kmh, mean_delta_deg, RMSE_DEG)
    wind_box = html.Div([
        html.Div([html.Strong("Level: "), html.Span(hazard_level)]),
        html.Div([html.Strong("Est wind (km/h): "), html.Span(f"{est_wind_kmh:.1f}")]),
        html.Div([html.Strong("Translation (km/h): "), html.Span(f"{translation_kmh:.1f}")])
    ], style={"backgroundColor":color,"padding":"10px","borderRadius":"6px","color":"black"})
    return fig, last_info, wind_box, f"{RMSE_DEG:.3f}", f"{RMSE_DEG*111:.0f} km"

if __name__ == "__main__":
    app.run(debug=True)
