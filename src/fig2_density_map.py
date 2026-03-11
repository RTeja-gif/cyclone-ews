import pandas as pd
import matplotlib.pyplot as plt

# Load cleaned data
df = pd.read_csv("data/processed/ibtracs_nio_cleaned.csv")
df.columns = [c.upper() for c in df.columns]

lat = df["LAT"].values
lon = df["LON"].values

plt.figure(figsize=(6,6))
plt.hist2d(lon, lat, bins=100, cmap="hot")
plt.colorbar(label="Track Density")

plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("Cyclone Track Density Map (North Indian Ocean)")
plt.grid(alpha=0.3)

plt.savefig("paper/figures/fig2_density_map.png", dpi=300)
plt.close()
