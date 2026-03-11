import os, sys, numpy as np, pandas as pd, joblib
from sklearn.preprocessing import StandardScaler

INPUT_CSV = "data/processed/ibtracs_nio_cleaned.csv"
OUT_PATH = "artifacts"
DEFAULT_SEQ_LEN = 6
PRED_STEPS = 1

os.makedirs(OUT_PATH, exist_ok=True)

if not os.path.exists(INPUT_CSV):
    print("ERROR: input file not found:", INPUT_CSV); sys.exit(1)
df = pd.read_csv(INPUT_CSV)
df.columns = [c.upper().strip() for c in df.columns]

print("Loaded file:", INPUT_CSV)
print("Rows,Cols:", df.shape)
print("Columns sample:", df.columns.tolist()[:20])

# detect columns
def find_col(candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

lat_col = find_col(["LAT","LATITUDE","LAT_DEG","LAT_DEG_N"])
lon_col = find_col(["LON","LONITUDE","LON_DEG","LON_DEG_E","LON_DEG"])
wind_col = find_col(["WMO_WIND","WIND","WIND_KTS","WIND_SPEED","MAX_WIND"])

print("Detected -> LAT:", lat_col, " LON:", lon_col, " WIND:", wind_col)

if lat_col is None or lon_col is None:
    print("ERROR: could not detect LAT or LON. Columns:", df.columns.tolist()); sys.exit(1)

# coerce numeric
df[lat_col] = pd.to_numeric(df[lat_col], errors="coerce")
df[lon_col] = pd.to_numeric(df[lon_col], errors="coerce")
if wind_col:
    df[wind_col] = pd.to_numeric(df[wind_col], errors="coerce")

# build features list only including wind if it has any non-null values
features = [lat_col, lon_col]
if wind_col:
    non_null_wind = int(df[wind_col].notna().sum())
    print("Non-null wind values:", non_null_wind)
    if non_null_wind > 0:
        features.append(wind_col)
    else:
        print("Wind column empty -> skipping wind feature.")

print("Using features:", features)

# drop rows missing core features
df = df.dropna(subset=features)
print("Rows after dropping missing feature rows:", len(df))
if len(df) == 0:
    print("ERROR: no rows remain after dropping NaNs for features. Exiting."); sys.exit(1)

# group column
group_col = None
for cand in ["SID","STORM_ID","NAME","TRACK_ID","ID","ID_NO","STORM"]:
    if cand in df.columns:
        group_col = cand; break
if group_col is None:
    if "SEASON" in df.columns and "NAME" in df.columns:
        df["SID"] = df["SEASON"].astype(str) + "_" + df["NAME"].astype(str)
        group_col = "SID"
    else:
        df["_GRP"] = 1
        group_col = "_GRP"

print("Using group column:", group_col)

# parse time if present
if "ISO_TIME" in df.columns:
    df["ISO_TIME"] = pd.to_datetime(df["ISO_TIME"], errors="coerce")

group_sizes = df.groupby(group_col).size().sort_values(ascending=False)
print("Top groups (count):\n", group_sizes.head(10).to_string())

max_len = int(group_sizes.max())
SEQ_LEN = DEFAULT_SEQ_LEN
if max_len < (DEFAULT_SEQ_LEN + PRED_STEPS):
    SEQ_LEN = max(2, max_len - PRED_STEPS)
    print("Short storms -> lowered SEQ_LEN to", SEQ_LEN)

# fit scaler
scaler = StandardScaler()
scaler.fit(df[features].to_numpy(dtype=float))
print("Fitted scaler.")

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
    print("No sequences created. Increase data or lower SEQ_LEN."); sys.exit(1)

X = np.array(X_list); y = np.array(y_list)
print("Raw sequences shapes:", X.shape, y.shape)

ns, sl, nf = X.shape
X_flat = X.reshape(-1, nf)
X_scaled = scaler.transform(X_flat).reshape(ns, sl, nf)

y_reshaped = y.reshape(ns * PRED_STEPS, nf)
y_scaled = scaler.transform(y_reshaped).reshape(ns, PRED_STEPS * nf)

np.savez_compressed(os.path.join(OUT_PATH, "sequences.npz"), X=X_scaled, y=y_scaled)
joblib.dump(scaler, os.path.join(OUT_PATH, "scaler.pkl"))

print("Saved sequences:", X_scaled.shape, y_scaled.shape)
print("Artifacts ->", OUT_PATH)
