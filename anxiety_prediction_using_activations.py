import os
import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error

def loo_cv_rmse(pipeline, X, y):
    """Compute leave‑one‑out RMSE for a given sklearn pipeline."""
    loo = LeaveOneOut()
    y_true, y_pred = [], []
    for train_idx, test_idx in loo.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train           = y[train_idx]
        pipeline.fit(X_train, y_train)
        y_true.append(y[test_idx][0])
        y_pred.append(pipeline.predict(X_test)[0])
    return np.sqrt(mean_squared_error(y_true, y_pred))

# --- Load data ---
df = pd.read_csv("data/cleaned_data.csv")
print(f"Loaded {len(df)} subjects from cleaned_data.csv")

 # --- Prepare X, y ---
drop_cols = ["id", "anxiety"]
X = df.drop(columns=drop_cols)
y = df["anxiety"].values

# --- Define pipelines ---
pipelines = {
    "Ordinary Least Squares": make_pipeline(
        StandardScaler(),
        LinearRegression()
    ),
    "Ridge (α via inner CV)": make_pipeline(
        StandardScaler(),
        RidgeCV(alphas=np.logspace(-3, 3, 31), cv=5)
    ),
    "LassoCV": make_pipeline(
        StandardScaler(),
        LassoCV(alphas=np.logspace(-3, 3, 31), cv=5, max_iter=5000)
    ),
}

# --- Run LOO CV for each ---
rmse_records = []
for name, pipe in pipelines.items():
    rmse = loo_cv_rmse(pipe, X, y)
    rmse_records.append({"model": name, "loo_rmse": rmse})
    print(f"{name} LOO‐RMSE: {rmse:.3f}")

os.makedirs("results", exist_ok=True)
rmse_df = pd.DataFrame(rmse_records)
rmse_df.to_csv("results/anxiety_prediction_rmse.csv", index=False)
print("\nSaved all LOO‐RMSEs to results/anxiety_prediction_rmse.csv\n")

# --- 2) Fit each model on the full dataset and save coefficients ---
for name, pipe in pipelines.items():
    pipe.fit(X, y)
    # Extract the final estimator from the pipeline
    # get whatever the last step is (OLS, RidgeCV, or LassoCV)
    estimator = pipe.steps[-1][1]
    coefs      = estimator.coef_
    intercept  = estimator.intercept_

    # Build a DataFrame of feature names + coefficients
    coef_df = pd.DataFrame({
        "feature": ["intercept"] + list(X.columns),
        "coef":    np.hstack([intercept, coefs])
    })

    coef_path = f"results/coefficients_{name.lower()}.csv"
    coef_df.to_csv(coef_path, index=False)
    print(f"{name} coefficients saved to {coef_path}")
    print(coef_df.head(5), "\n")