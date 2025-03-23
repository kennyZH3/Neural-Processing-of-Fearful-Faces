import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind

df = pd.read_csv("data/cleaned_data.csv")
print(df.head())

anxiety_distribution = df['anxiety'].value_counts()
# Calculate mean, median, and standard deviation
mean_anxiety = df['anxiety'].mean()
median_anxiety = df['anxiety'].median()
std_anxiety = df['anxiety'].std()

# Print the statistics
print("Mean Anxiety Score:", mean_anxiety)
print("Median Anxiety Score:", median_anxiety)
print("Standard Deviation of Anxiety Score:", std_anxiety)

# Visualize the distribution
plt.hist(df['anxiety'], bins=10)
plt.xlabel('Anxiety Score')
plt.ylabel('Frequency')
plt.title('Distribution of Anxiety Scores')
plt.savefig("plots/anxiety_distribution.png", dpi=300)
plt.show()

plt.figure()
plt.boxplot(df['anxiety'], vert=False)
plt.xlabel('Mean Anxiety Score')
plt.title('Boxplot of Participant Mean Anxiety Scores')
plt.savefig("plots/anxiety_boxplot.png", dpi=300)  
plt.show()

# Filter anxiety scores to create a binary variable with balanced classes
median_val = df['anxiety'].median()
df['anxiety_binary'] = df['anxiety'].apply(lambda x: 'high' if x >= median_val else 'low')
class_counts = df['anxiety_binary'].value_counts()
print("Class counts:\n", class_counts)
num_variables = len(df.columns)
num_observations = len(df)
print(f"Number of variables: {num_variables}")
print(f"Number of observations: {num_observations}")

# Pivot to long format 
df_long = df.melt(
    id_vars=["id", "anxiety", "anxiety_binary"],
    var_name="condition_roi",
    value_name="activation"
).assign(
    condition=lambda x: x.condition_roi.str.split("__").str[0],
    region=lambda x: x.condition_roi.str.split("__").str[1]
).drop(columns="condition_roi")

print(df_long.head())

num_variables = df_long.shape[1]
num_observations = df_long.shape[0]
print(f"Number of variables: {num_variables}")
print(f"Number of observations: {num_observations}")


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