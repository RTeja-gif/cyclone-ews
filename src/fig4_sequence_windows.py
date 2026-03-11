import pandas as pd
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("data/processed/ibtracs_nio_cleaned.csv")
df.columns = [c.upper() for c in df.columns]

# Select one long storm
sid = df["SID"].value_counts().idxmax()
g = df[df["SID"] == sid].sort_values("ISO_TIME")

lat = g["LAT"].values
lon = g["LON"].values

SEQ_LEN = 12

plt.figure(figsize=(6,6))

# Plot full track
plt.plot(lon, lat, color="lightgray", linewidth=2, label="Full Track")

# Highlight one sequence window
plt.plot(
    lon[:SEQ_LEN],
    lat[:SEQ_LEN],
    marker="o",
    color="red",
    linewidth=2,
    label=f"LSTM Sequence (length={SEQ_LEN})"
)

plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("Sequence Window Construction for LSTM")
plt.legend()
plt.grid(alpha=0.3)

plt.savefig("paper/figures/fig4_sequence_windows.png", dpi=300)
plt.close()
