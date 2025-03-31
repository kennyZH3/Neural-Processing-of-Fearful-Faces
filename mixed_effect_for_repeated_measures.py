import pandas as pd
import statsmodels.formula.api as smf

# Read in the long format data
df = pd.read_csv("data/long_data.csv")
print(len(df))
df.head()

# Set up dictionary to store results
model_results = {}

# Fit a mixed effects model for repeated measurement for each region
for region in df['region'].unique():
    # Subset the data for the current region
    df_region = df[df['region'] == region].copy()
    
    # Ensure that 'condition' is treated as a categorical variable
    df_region['condition'] = df_region['condition'].astype('category')
    
    # Fit the mixed effects model
    # activation ~ anxiety + condition + anxiety:condition, with random intercept for each subject (id)
    model = smf.mixedlm("activation ~ anxiety * condition", data=df_region, groups=df_region["id"])
    result = model.fit()
    
    # Store the result in the dictionary
    model_results[region] = result
    
    # Print a summary for this region
    print(f"Region: {region}")
    print(result.summary())
    print("\n" + "="*80 + "\n")

    # Save the summary to a text file
    with open(f"results/results_{region}.txt", "w") as f:
        f.write(result.summary().as_text())