# Anxiety‑Activation Analysis

This repository investigates how participants' anxiety scores influence brain activation under different task conditions using neuroimaging data. 
We process the raw data, perform statistical comparisons, fit mixed‑effects models, and build predictive models to explore anxiety‑activation relationships.

## Usage

1. Preprocessing

```bash
python 01_data_preprocessing.py
```

Outputs: data/cleaned_data.csv, data/long_data.csv, and initial plots in plots/.

2. Method 1: Hypothesis Testing

```bash
python 02_method_1_mean_hypo_test.py
```

Performs independent‑samples t‑tests across conditions & regions.

Saves results to results/hypothesis_test_results.csv.

3. Method 2: Mixed‑Effects Modeling

```bash
python 02_method_2_mixed_effect_for_repeated_measures.py
```

Fits one mixed model per ROI: activation ~ anxiety * condition, with random intercepts by subject.

Saves per‑region summaries and LOPO‑CV RMSE to results/all_regions_rmse.csv.

4. Method 3: Predictive Modeling

```bash
python 02_method_3_anxiety_prediction_using_activations.py
```
Uses activation features to predict continuous anxiety via LOO‑CV.

Compares OLS, RidgeCV, and LassoCV models.

Saves RMSEs to results/anxiety_prediction_rmse.csv and model coefficients to results/coefficients_*.csv.