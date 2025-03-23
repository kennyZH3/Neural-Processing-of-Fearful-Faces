import pandas as pd
from scipy.stats import ttest_ind

df_long = pd.read_csv("data/long_data.csv")
print(df_long.head())


# Hypothesis testing: Compare activation levels between high and low anxiety groups for each condition and region
results = []

# Loop through each condition × region
for (cond, region), subset in df_long.groupby(['condition', 'region']):
    high_vals = subset.loc[subset.anxiety_binary == 'high', 'activation']
    low_vals  = subset.loc[subset.anxiety_binary == 'low',  'activation']
    
    if len(high_vals) >= 2 and len(low_vals) >= 2:
        t_stat, p_val = ttest_ind(high_vals, low_vals, equal_var=False)
        results.append({
            'condition': cond,
            'region': region,
            'n_high': len(high_vals),
            'n_low': len(low_vals),
            'mean_high': high_vals.mean(),
            'mean_low': low_vals.mean(),
            't_stat': t_stat,
            'p_value': p_val,
            'significant': p_val < 0.05
        })

results_df = pd.DataFrame(results)

# Sort by p‑value
results_df = results_df.sort_values('p_value').reset_index(drop=True)

print(results_df)

# Print summary statistics
class_counts = results_df['significant'].value_counts()
print("Class counts:\n", class_counts)
# Print regions with significant differences
diff_regions = results_df[results_df['significant'] == True]
print("Regions with significant differences undet different conditions:\n", diff_regions['condition'] + " " + diff_regions['region'])

# Save the results to a CSV file
results_df.to_csv("data/hypothesis_test_results.csv", index=False)

# Save the long format data to a CSV file
df_long.to_csv("data/long_data.csv", index=False)