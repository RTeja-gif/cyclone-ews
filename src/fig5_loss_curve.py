import json
import matplotlib.pyplot as plt

with open("artifacts/training_history.json") as f:
    h = json.load(f)

plt.plot(h["loss"], label="Training Loss")
plt.plot(h["val_loss"], label="Validation Loss")

plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("LSTM Training and Validation Loss")
plt.legend()
plt.grid(alpha=0.3)

plt.savefig("paper/figures/fig5_loss_curve.png", dpi=300)
plt.close()
