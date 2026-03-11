import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/processed/ibtracs_nio_cleaned.csv")
df.columns = [c.upper() for c in df.columns]

storms = df["SID"].unique()[:3]

plt.figure(figsize=(6,6))
for sid in storms:
    g = df[df["SID"] == sid]
    plt.plot(g["LON"], g["LAT"], marker='o', label=sid)

plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("Raw Cyclone Tracks (Sample)")
plt.legend()
plt.grid()

plt.savefig("paper/figures/fig1_raw_tracks.png", dpi=300)
plt.close()
