import pandas as pd
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("data/processed/ibtracs_nio_cleaned.csv")
df.columns = [c.upper() for c in df.columns]

# Pick 9 different storms
storm_ids = df["SID"].unique()[:9]

plt.figure(figsize=(9,9))

for i, sid in enumerate(storm_ids, 1):
    g = df[df["SID"] == sid]
    plt.subplot(3, 3, i)
    plt.plot(g["LON"], g["LAT"], linewidth=1.5)
    plt.title(sid, fontsize=8)
    plt.xticks([])
    plt.yticks([])
    plt.grid(alpha=0.3)

plt.suptitle("Diversity of Cyclone Tracks", fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.96])

plt.savefig("paper/figures/fig3_track_diversity.png", dpi=300)
plt.close()
