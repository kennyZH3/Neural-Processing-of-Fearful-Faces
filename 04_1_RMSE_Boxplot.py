import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1) Load and sort by RMSE
df = pd.read_csv("results/all_regions_rmse.csv")  # adjust path if needed
df = df.sort_values("rmse")

# 2) Make sure the plots folder exists
os.makedirs("plots", exist_ok=True)

# 3) Compute x positions
x = np.arange(len(df))
width = 0.35

# 4) Plot side-by-side bars
plt.figure(figsize=(12, 6))
plt.bar(x - width/2, df["rmse"], width, label="LOSO-CV RMSE")
plt.bar(x + width/2, df["activation_sd"], width, label="Activation SD")

# 5) Formatting
plt.xticks(x, df["region"], rotation=45, ha="right")
plt.xlabel("Brain Region")
plt.ylabel("Value")
plt.title("LOSO-CV RMSE vs Activation SD by Region")
plt.legend()
plt.tight_layout()

# 6) Save and show
plt.savefig("plots/rmse_vs_sd_by_region.png", dpi=300)
plt.show()
