import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("results.csv")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Loss
ax1.plot(df["epoch"], df["train_loss"], "o-", label="Train Loss", color="#2196F3")
ax1.plot(df["epoch"], df["val_loss"], "o-", label="Val Loss", color="#F44336")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss")
ax1.set_title("Train vs Val Loss")
ax1.set_xticks(df["epoch"])
ax1.legend()
ax1.grid(True, alpha=0.3)

# AUC
ax2.plot(df["epoch"], df["val_auc"], "o-", label="Val AUC", color="#4CAF50")
ax2.axhline(y=df["val_auc"].max(), linestyle="--", color="gray", alpha=0.5,
            label=f"Best AUC: {df['val_auc'].max():.4f} (epoch {df['val_auc'].idxmax() + 1})")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("AUC")
ax2.set_title("Validation AUC")
ax2.set_xticks(df["epoch"])
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("training_curves.png", dpi=150)
plt.show()
print(f"Saved to training_curves.png")
