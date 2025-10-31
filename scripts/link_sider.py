import os
import pandas as pd
from tqdm import tqdm

# --- Step 1: Load Data ---
script_dir = os.path.dirname(os.path.abspath(__file__))

n2c2_path = os.path.join(script_dir, "..", "data", "n2c2", "processed", "n2c2_clean.csv")
sider_path = os.path.join(script_dir, "..", "data", "sider", "processed", "sider_clean.csv")

n2c2_df = pd.read_csv(n2c2_path)
sider_df = pd.read_csv(sider_path)

print(f"Loaded {len(n2c2_df)} ADE-Drug pairs from n2c2.")
print(f"Loaded {len(sider_df)} entries from SIDER.")


# --- Step 3: Define Matching Function ---
def check_sider_match(drug, ade, sider_df):
    """
    Check if a drug-ADE pair exists in the SIDER dataset.
    Returns (match_found: bool, frequency: str or None)
    """
    matches = sider_df[
        (sider_df["drug_norm"] == drug) & 
        (sider_df["ade_norm"] == ade)
    ]
    if not matches.empty:
        freq = matches["frequency"].iloc[0] if "frequency" in matches.columns else None
        return True, freq
    return False, None

# --- Step 4: Apply Matching Across n2c2 Data ---
in_sider_list = []
freq_list = []

print("\nMatching n2c2 ADE-Drug pairs with SIDER...")
for _, row in tqdm(n2c2_df.iterrows(), total=len(n2c2_df)):
    in_sider, freq = check_sider_match(row["drug_norm"], row["ade_norm"], sider_df)
    in_sider_list.append(in_sider)
    freq_list.append(freq)

n2c2_df["in_SIDER"] = in_sider_list
n2c2_df["frequency"] = freq_list

# --- Step 5: Save Output ---
output_dir = os.path.join(script_dir, "..", "data", "n2c2", "processed")
output_path = os.path.join(output_dir, "n2c2_with_sider_context.csv")

n2c2_df.to_csv(output_path, index=False)
print(f"\nSaved file with SIDER context: {output_path}")
print(f"Total pairs matched in SIDER: {n2c2_df['in_SIDER'].sum()}")

