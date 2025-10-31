import os, re, pandas as pd
import glob


script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, "..", "data", "n2c2", "processed")

# Input and output file paths
input_file = os.path.join(data_dir, "n2c2_normalized.csv")
output_file = os.path.join(data_dir, "n2c2_clean.csv")

# Read the CSV
df = pd.read_csv(input_file)

# Create new columns with priority logic:
# use matched if available, else normalized
df["drug_norm"] = df["drug_matched"].fillna(df["drug_normalized"])
df["ade_norm"] = df["ade_matched"].fillna(df["ade_normalized"])

# In some cases, there may be empty strings instead of NaN
df["drug_norm"] = df["drug_norm"].replace("", pd.NA).fillna(df["drug_normalized"])
df["ade_norm"] = df["ade_norm"].replace("", pd.NA).fillna(df["ade_normalized"])

# Keep only required columns
result_df = df[["drug_norm", "ade_norm", "source_file"]]

# Save to new CSV
result_df.to_csv(output_file, index=False)

print(f"New file saved as: {output_file}")
