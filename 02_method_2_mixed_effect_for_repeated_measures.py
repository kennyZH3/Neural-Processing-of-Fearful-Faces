import os
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from sklearn.model_selection import LeaveOneGroupOut

# --- Setup paths and results folder ---
DATA_PATH   = "data/long_data.csv"
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# --- Read data ---
df = pd.read_csv(DATA_PATH)
print(f"Loaded {len(df)} rows from {DATA_PATH}")

# --- Prepare Leave-One-Subject-Out splitter ---
logo = LeaveOneGroupOut()

# Will accumulate one dict per region
rmse_records = []

# --- Loop over regions ---
for region in df['region'].unique():
    print(f"Processing region: {region}")
    df_region = df[df['region'] == region].copy()
    df_region['condition'] = df_region['condition'].astype('category')
    
    # compute region-wide activation sd
    activation_sd = df_region['activation'].std(ddof=1)
    
    # --- 1) Fit full model and save summary ---
    full_mod   = smf.mixedlm("activation ~ anxiety * condition",
                             data=df_region,
                             groups=df_region["id"])
    full_res   = full_mod.fit(reml=False)
    
    summary_txt  = full_res.summary().as_text()
    summary_path = os.path.join(RESULTS_DIR, f"model_summary_{region}.txt")
    with open(summary_path, "w") as f:
        f.write(summary_txt)
    print(f"  • Full-data summary saved to {summary_path}")
    
    # --- 2) Leave-one-subject-out CV for RMSE ---
    fold_mses = []
    for train_idx, test_idx in logo.split(df_region, groups=df_region["id"]):
        train_df = df_region.iloc[train_idx]
        test_df  = df_region.iloc[test_idx]
        
        cv_mod = smf.mixedlm("activation ~ anxiety * condition",
                             data=train_df,
                             groups=train_df["id"])
        cv_res = cv_mod.fit(reml=False)
        
        y_true = test_df["activation"].values
        y_pred = cv_res.predict(test_df).values
        fold_mses.append(np.mean((y_true - y_pred)**2))
    
    rmse = np.sqrt(np.mean(fold_mses))
    
    # record RMSE and activation SD
    rmse_records.append({
        "region":       region,
        "rmse":         rmse,
        "activation_sd": activation_sd
    })
    print(f"  • LOPO CV RMSE = {rmse:.3f}, activation SD = {activation_sd:.3f}\n")

# --- Save combined RMSEs + SDs ---
rmse_df = pd.DataFrame(rmse_records).sort_values("region")
rmse_csv = os.path.join(RESULTS_DIR, "all_regions_rmse.csv")
rmse_df.to_csv(rmse_csv, index=False)
print(f"All regions RMSE & SD saved to {rmse_csv}")
