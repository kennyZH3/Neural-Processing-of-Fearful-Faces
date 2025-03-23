import pandas as pd

# Step 1: Load the data
df = pd.read_excel("data/original_data.xlsx", sheet_name=4)
print(df.head())

num_variables = len(df.columns)
num_observations = len(df)
print(f"Number of variables: {num_variables}")
print(f"Number of observations: {num_observations}")

# Step 2: Clean the variable names according to the mapping
cond_map = {
    "HighLoadFearful":           "highload_fearful",
    "HighLoadNeutral":           "highload_neutral",
    "LowLoadFearful":            "lowload_fearful",
    "LowLoadNeutral":            "lowload_neutral",
    "HighLoadVsLowLoad":         "high_vs_low_load",
    "HighLoadFearfulVsHighLoadNeutral": "high_fearful_vs_high_neutral",
    "LowLoadFearfulVsLowLoadNeutral":   "low_fearful_vs_low_neutral",
    "HighLoadFearfulVsLowLoadFearful":  "high_fearful_vs_low_fearful",
    "HighLoadNeutralVsLowLoadNeutral":  "high_neutral_vs_low_neutral",
    "FearfulVsNeutral":                  "fearful_vs_neutral",
}

roi_map = {
    "Dorsal_Anterior_Cingulate_Cortex_8mm_sphere_mask_cope1": "dacc",
    "Left_DLPFC_8mm_sphere_mask_cope1":                     "ldlpfc",
    "Left_STS_8mm_sphere_mask_cope1":                       "lsts",
    "Left_VLPFC_8mm_sphere_mask_cope1":                     "lvlpfc",
    "Right_DLPFC_8mm_sphere_mask_cope1":                    "rdlpfc",
    "Right_STS_8mm_sphere_mask_cope1":                      "rsts",
    "Right_VLPFC_8mm_sphere_mask_cope1":                    "rvlpfc",
    "Rostral_Anterior_Cingulate_Cortex_8mm_sphere_mask_cope1": "racc",
    "Left_Amygdala_cope1":                                  "lamyg",
    "Right_Amygdala_cope1":                                 "ramyg",
    "Left_Insula_cope1":                                    "linsula",
    "Right_Insula_cope1":                                   "rinsula",
    "Right_DorsalACC_cope1":                                "rdacc",
    "Right_RostralACC_cope1":                               "rracc",
    "Left_RostralACC_cope1":                                "lracc",
    "Left_DorsalACC_cope1":                                 "ldacc",
}

rename_map = {}
for col in df.columns:
    if col not in {"Participant", "Trait Anxiety Score"}:
        for cond, cond_short in cond_map.items():
            if col.startswith(cond + "_"):
                roi_key = col.replace(cond + "_", "")
                if roi_key in roi_map:
                    rename_map[col] = f"{cond_short}__{roi_map[roi_key]}"
                break

df = df.rename(columns=rename_map)

df = df.rename(columns={
    "Participant": "id",
    "Trait Anxiety Score": "anxiety"
})


# Step 4: Saving the new data
df.to_csv("data/cleaned_data.csv", index=False)