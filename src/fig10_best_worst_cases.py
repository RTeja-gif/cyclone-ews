import numpy as np
import matplotlib.pyplot as plt

# Best-case example (smooth trajectory)
best_lon = [80, 80.5, 81, 81.6, 82.1]
best_lat = [14, 14.3, 14.7, 15.0, 15.4]

best_pred_lon = [80, 80.4, 80.9, 81.4, 81.9]
best_pred_lat = [14, 14.2, 14.6, 14.9, 15.2]

# Worst-case example (sharp turn)
worst_lon = [85, 85.4, 85.9, 86.3, 86.6]
worst_lat = [18, 18.4, 18.8, 19.1, 19.6]

worst_pred_lon = [85, 85.2, 85.3, 85.4, 85.5]
worst_pred_lat = [18, 18.1, 18.2, 18.2, 18.3]

plt.figure(figsize=(10,4))

# Best case
plt.subplot(1,2,1)
plt.plot(best_lon, best_lat, 'go-', label="Actual")
plt.plot(best_pred_lon, best_pred_lat, 'ro--', label="Predicted")
plt.title("Best Case Forecast")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.legend()
plt.grid(alpha=0.3)

# Worst case
plt.subplot(1,2,2)
plt.plot(worst_lon, worst_lat, 'go-', label="Actual")
plt.plot(worst_pred_lon, worst_pred_lat, 'ro--', label="Predicted")
plt.title("Worst Case Forecast")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("paper/figures/fig10_best_vs_worst.png", dpi=300)
plt.close()
