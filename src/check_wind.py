import pandas as pd
import numpy as np

df = pd.read_csv("data/processed/ibtracs_nio_cleaned.csv")
df.columns = [c.upper().strip() for c in df.columns]

print("Columns sample:", df.columns.tolist()[:40])

wind_col = None
for c in ["WMO_WIND","WIND","WIND_KTS","WIND_SPEED","MAX_WIND"]:
    if c in df.columns:
        wind_col = c
        break

if wind_col:
    non_null = int(df[wind_col].notna().sum())
    print(f"Found wind column: {wind_col}  non-null count: {non_null} / {len(df)}")
else:
    print("No wind column found in dataset.")

# compute simple translation speed for the largest storm (if ISO_TIME exists)
group_col = None
for cand in ["SID","STORM_ID","NAME","TRACK_ID","ID"]:
    if cand in df.columns:
        group_col = cand
        break

if group_col is None:
    if "SEASON" in df.columns and "NAME" in df.columns:
        df["SID"] = df["SEASON"].astype(str) + "_" + df["NAME"].astype(str)
        group_col = "SID"
    else:
        group_col = None

if group_col:
    if "ISO_TIME" in df.columns:
        df["ISO_TIME"] = pd.to_datetime(df["ISO_TIME"], errors="coerce")
    groups = df.groupby(group_col).size().sort_values(ascending=False)
    top = groups.index[0]
    g = df[df[group_col]==top].sort_values("ISO_TIME") if "ISO_TIME" in df.columns else df[df[group_col]==top]
    print("\\nTop storm group:", top, "points:", len(g))
    if len(g) >= 2:
        a = g.iloc[-2]
        b = g.iloc[-1]
        lat1, lon1 = float(a["LAT"]), float(a["LON"])
        lat2, lon2 = float(b["LAT"]), float(b["LON"])
        avg_lat = np.deg2rad((lat1+lat2)/2.0)
        dx = (lon2 - lon1) * 111.320 * np.cos(avg_lat)
        dy = (lat2 - lat1) * 110.574
        dist_km = float(np.sqrt(dx*dx + dy*dy))
        time_info = "N/A"
        speed_kph = None
        if "ISO_TIME" in g.columns and not pd.isna(a["ISO_TIME"]) and not pd.isna(b["ISO_TIME"]):
            dt = (b["ISO_TIME"] - a["ISO_TIME"]).total_seconds() / 3600.0
            if dt > 0:
                speed_kph = dist_km / dt
                time_info = f"{dt:.2f} hours"
        print(f"Last displacement between last two obs: {dist_km:.2f} km (time: {time_info})")
        if speed_kph:
            print(f"Estimated translation speed: {speed_kph:.2f} km/h")
        else:
            print("Could not compute speed (missing/invalid ISO_TIME).")
    else:
        print("Not enough points to compute translation speed for top storm.")
else:
    print("No group column available to compute speeds.")
