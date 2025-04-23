import os
import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import LeaveOneOut
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error

def loo_cv_rmse(pipeline, X, y):
    """Compute leave-one-out RMSE for a given sklearn pipeline."""
    loo = LeaveOneOut()
    y_true, y_pred = [], []
    for train_idx, test_idx in loo.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train = y[train_idx]
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

# --- Define pipelines on original features ---
pipelines = {
    "OLS":   make_pipeline(StandardScaler(), LinearRegression()),
    "Ridge": make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 31), cv=5)),
    "Lasso": make_pipeline(StandardScaler(), LassoCV(alphas=np.logspace(-3, 3, 31), cv=5, max_iter=5000)),
    "RF":    make_pipeline(StandardScaler(), RandomForestRegressor(n_estimators=100, random_state=42)),
}

# --- Define pipelines on PCA-reduced features (retain 95% variance) ---
pipelines_pca = {
    "OLS_PCA":   make_pipeline(StandardScaler(), PCA(n_components=0.95), LinearRegression()),
    "Ridge_PCA": make_pipeline(StandardScaler(), PCA(n_components=0.95), RidgeCV(alphas=np.logspace(-3, 3, 31), cv=5)),
    "Lasso_PCA": make_pipeline(StandardScaler(), PCA(n_components=0.95), LassoCV(alphas=np.logspace(-3, 3, 31), cv=5, max_iter=5000)),
    "RF_PCA":    make_pipeline(StandardScaler(), PCA(n_components=0.95), RandomForestRegressor(n_estimators=100, random_state=42)),
}

os.makedirs("results", exist_ok=True)

# --- 1) Evaluate all models via LOO-CV ---
all_rmse = []

print("\n=== Original Features ===")
for name, pipe in pipelines.items():
    rmse = loo_cv_rmse(pipe, X, y)
    all_rmse.append({"model": name, "type": "original", "loo_rmse": rmse})
    print(f"{name:6s} LOO-RMSE: {rmse:.3f}")

print("\n=== PCA-Reduced Features ===")
for name, pipe in pipelines_pca.items():
    rmse = loo_cv_rmse(pipe, X, y)
    all_rmse.append({"model": name, "type": "pca", "loo_rmse": rmse})
    print(f"{name:8s} LOO-RMSE: {rmse:.3f}")

# Save comparison CSV
rmse_df = pd.DataFrame(all_rmse)
rmse_df.to_csv("results/anxiety_prediction_rmse_comparison.csv", index=False)
print("\nSaved all LOO-RMSEs to results/anxiety_prediction_rmse_comparison.csv\n")

# --- 2) Fit each model on the full dataset and save coefficients/importances ---
for name, pipe in {**pipelines, **pipelines_pca}.items():
    pipe.fit(X, y)
    estimator = pipe.steps[-1][1]

    if isinstance(estimator, RandomForestRegressor):
        # Save feature importances
        if "PCA" in name:
            pca = pipe.named_steps["pca"]
            feat_names = [f"PC{i+1}" for i in range(pca.n_components_)]
        else:
            feat_names = list(X.columns)

        importances = estimator.feature_importances_
        fi_df = pd.DataFrame({
            "feature": feat_names,
            "importance": importances
        }).sort_values("importance", ascending=False)

        fi_path = f"results/feature_importances_{name.lower()}.csv"
        fi_df.to_csv(fi_path, index=False)
        print(f"{name} feature importances saved to {fi_path}")
        print(fi_df.head(5), "\n")

    else:
        # For linear models, extract coefficients
        if "PCA" in name:
            pca = pipe.named_steps["pca"]
            coef_names = [f"PC{i+1}" for i in range(pca.n_components_)]
        else:
            coef_names = list(X.columns)

        coefs = estimator.coef_
        intercept = estimator.intercept_

        coef_df = pd.DataFrame({
            "feature": ["intercept"] + coef_names,
            "coef":    np.hstack([intercept, coefs])
        })
        coef_path = f"results/coefficients_{name.lower()}.csv"
        coef_df.to_csv(coef_path, index=False)
        print(f"{name} coefficients saved to {coef_path}")
        print(coef_df.head(5), "\n")
