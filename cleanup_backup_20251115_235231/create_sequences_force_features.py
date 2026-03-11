import os, sys, numpy as np, pandas as pd, joblib
from sklearn.preprocessing import StandardScaler

INPUT_CSV = "data/processed/ibtracs_nio_cleaned.csv"
OUT_PATH = "artifacts"
SEQ_LEN = 12
PRED_STEPS = 1

os.makedirs(OUT_PATH, exist_ok=True)
if not os.path.exists(INPUT_CSV):
    print("ERROR: input file not found:", INPUT_CSV); sys.exit(1)

df = pd.read_csv(INPUT_CSV)
df.columns = [c.upper().strip() for c in df.columns]

# helpers
def find_col(candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

lat_col = find_col(["LAT","LATITUDE"])
lon_col = find_col(["LON","LONGITUDE","LON"])
press_col = find_col(["WMO_PRES","PRESSURE","PRES","P"])
wind_col = find_col(["WMO_WIND","WIND","WIND_SPEED","MAX_WIND"])

print("Detected -> LAT:", lat_col, " LON:", lon_col, " PRES:", press_col, " WIND:", wind_col)

if lat_col is None or lon_col is None:
    print("ERROR: LAT/LON missing"); sys.exit(1)

# coerce numeric
for c in [lat_col, lon_col, press_col, wind_col]:
    if c and c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

# fill missing pressure and wind with median if present
features = [lat_col, lon_col]
if press_col and press_col in df.columns:
    med = df[press_col].median()
    df[press_col].fillna(med, inplace=True)
    features.append(press_col)
    print("Filled PRESS with median:", med)
if wind_col and wind_col in df.columns:
    medw = df[wind_col].median()
    df[wind_col].fillna(medw, inplace=True)
    if wind_col not in features:
        features.append(wind_col)
    print("Filled WIND with median:", medw)

print("Using features:", features)

# drop rows missing core lat/lon
df = df.dropna(subset=[lat_col, lon_col])
print("Rows after dropping missing lat/lon:", len(df))

# group column
group_col = None
for cand in ["SID","STORM_ID","NAME","TRACK_ID","ID"]:
    if cand in df.columns:
        group_col = cand; break
if group_col is None:
    if "SEASON" in df.columns and "NAME" in df.columns:
        df["SID"] = df["SEASON"].astype(str) + "_" + df["NAME"].astype(str)
        group_col = "SID"
    else:
        df["_GRP"] = 1
        group_col = "_GRP"

if "ISO_TIME" in df.columns:
    df["ISO_TIME"] = pd.to_datetime(df["ISO_TIME"], errors="coerce")

group_sizes = df.groupby(group_col).size().sort_values(ascending=False)
print("Top groups (count):\n", group_sizes.head(8).to_string())

# recompute SEQ length dynamically if needed
max_len = int(group_sizes.max()) if len(group_sizes)>0 else 0
if max_len < SEQ_LEN + PRED_STEPS:
    SEQ_LEN = max(2, max_len - PRED_STEPS)
    print("Lowered SEQ_LEN to", SEQ_LEN)

# fit scaler
scaler = StandardScaler()
scaler.fit(df[features].to_numpy(dtype=float))
print("Fitted scaler on features.")

# build sequences
X_list, y_list = [], []
for gid, group in df.groupby(group_col):
    g = group.sort_values("ISO_TIME") if "ISO_TIME" in group.columns else group
    arr = g[features].to_numpy(dtype=float)
    if len(arr) < SEQ_LEN + PRED_STEPS:
        continue
    for i in range(0, len(arr) - SEQ_LEN - PRED_STEPS + 1):
        seq = arr[i:i+SEQ_LEN]
        target = arr[i+SEQ_LEN:i+SEQ_LEN+PRED_STEPS]
        last = seq[-1]
        delta = (target - last).reshape(-1)
        X_list.append(seq)
        y_list.append(delta)

if len(X_list) == 0:
    print("No sequences created. Exiting."); sys.exit(1)

X = np.array(X_list); y = np.array(y_list)
print("Raw sequences shapes:", X.shape, y.shape)

ns, sl, nf = X.shape
X_flat = X.reshape(-1, nf)
X_scaled = scaler.transform(X_flat).reshape(ns, sl, nf)
y_reshaped = y.reshape(ns * PRED_STEPS, nf)
y_scaled = scaler.transform(y_reshaped).reshape(ns, PRED_STEPS * nf)

np.savez_compressed(os.path.join(OUT_PATH, "sequences_forced.npz"), X=X_scaled, y=y_scaled)
joblib.dump(scaler, os.path.join(OUT_PATH, "scaler_forced.pkl"))

print("Saved sequences_forced:", X_scaled.shape, y_scaled.shape)
