#!/usr/bin/env python3

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

def load_data():
    """
    Load the necessary data files from n2c2 and SIDER.
    """
    base_path = Path(__file__).parent.parent
    n2c2_path = base_path / 'data' / 'n2c2' / 'processed'
    sider_path = base_path / 'data' / 'sider' / 'processed'
    
    # Load the cleaned n2c2 data (contains normalized drug and ADE terms)
    # columns expected: drug_norm, ade_norm, source_file
    n2c2_normalized = pd.read_csv(n2c2_path / 'n2c2_clean.csv')
    
    # Load SIDER side effects data
    sider_effects = pd.read_csv(sider_path / 'sider_clean.csv')
    
    return n2c2_normalized, sider_effects

def check_ade_consistency(n2c2_data, sider_data):
    """
    Check if the ADEs in n2c2 data are consistent with SIDER's documented side effects.
    """
    # Prepare SIDER lookup sets (lowercased, stripped)
    sider_data = sider_data.fillna("")
    sider_drug = sider_data['drug_norm'].astype(str).str.lower().str.strip()
    sider_ade = sider_data['ade_norm'].astype(str).str.lower().str.strip()
    sider_pairs = set(zip(sider_drug, sider_ade))
    sider_drugs_set = set(sider_drug.unique())

    # Normalize n2c2 data fields we'll use
    n2c2_data = n2c2_data.copy()
    # expect columns: drug_norm, ade_norm
    n2c2_data['drug_norm'] = n2c2_data['drug_norm'].astype(str).str.lower().str.strip()
    n2c2_data['ade_norm'] = n2c2_data['ade_norm'].astype(str).str.lower().str.strip()

    # Add columns to track consistency
    n2c2_data['is_consistent'] = False
    n2c2_data['sider_match_found'] = False

    # Check each n2c2 entry against SIDER
    for idx, row in n2c2_data.iterrows():
        drug = row['drug_norm']
        ade = row['ade_norm']

        # Check if the drug-ADE pair exists in SIDER
        if (drug, ade) in sider_pairs:
            n2c2_data.at[idx, 'is_consistent'] = True
            n2c2_data.at[idx, 'sider_match_found'] = True
        else:
            # Check if the drug exists in SIDER with any side effect
            drug_exists = drug in sider_drugs_set
            n2c2_data.at[idx, 'sider_match_found'] = drug_exists
    
    return n2c2_data

def generate_validation_report(validated_data):
    """
    Generate a report summarizing the ADE validation results.
    """
    total_ades = len(validated_data)
    consistent_ades = validated_data['is_consistent'].sum()
    drugs_in_sider = validated_data['sider_match_found'].sum()
    
    # Calculate statistics
    consistency_rate = (consistent_ades / total_ades) * 100
    coverage_rate = (drugs_in_sider / total_ades) * 100
    
    # Group results by drug to analyze patterns (n2c2 uses 'drug_norm')
    drug_stats = validated_data.groupby('drug_norm').agg({
        'is_consistent': ['count', 'sum'],
        'sider_match_found': 'sum'
    }).round(2)
    
    # Save detailed results (add top-of-file comments describing purpose)
    output_path = Path(__file__).parent.parent / 'data' / 'n2c2' / 'processed'
    output_path.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().isoformat()

    # Write enriched n2c2 entries with SIDER context
    n2c2_out = output_path / 'n2c2_with_sider_context.csv'
    # open with newline='' to avoid extra blank lines on Windows when pandas writes CSV
    with open(n2c2_out, 'w', encoding='utf-8', newline='') as fh:
        # fh.write(f"# File: n2c2_with_sider_context.csv\n")
        # fh.write(f"# Purpose: n2c2 entries enriched with SIDER validation flags.\n")
        # fh.write(f"# Columns: original n2c2 fields plus 'is_consistent' (exact drug-ADE in SIDER) and 'sider_match_found' (drug exists in SIDER).\n")
        # fh.write(f"# Generated: {ts}\n")
        validated_data.to_csv(fh, index=False)

    # Make drug-level stats clearer and write with comments
    # Flatten and rename aggregated columns for clarity
    drug_stats = validated_data.groupby('drug_norm').agg(
        n_ades=('is_consistent', 'count'),
        n_consistent=('is_consistent', 'sum'),
        n_drugs_in_sider=('sider_match_found', 'sum')
    ).reset_index()

    stats_out = output_path / 'drug_validation_stats.csv'
    # open with newline='' to avoid extra blank lines on Windows when pandas writes CSV
    with open(stats_out, 'w', encoding='utf-8', newline='') as fh:
        # fh.write(f"# File: drug_validation_stats.csv\n")
        # fh.write(f"# Purpose: Per-drug summary of ADE validation against SIDER.\n")
        # fh.write(f"# Columns: drug_norm (normalized drug name), n_ades (number of ADE mentions), n_consistent (number of mentions whose ADE matched SIDER), n_drugs_in_sider (number of mentions where the drug exists in SIDER).\n")
        # fh.write(f"# Generated: {ts}\n")
        drug_stats.to_csv(fh, index=False)
    
    return {
        'total_ades': total_ades,
        'consistent_ades': consistent_ades,
        'consistency_rate': consistency_rate,
        'coverage_rate': coverage_rate
    }

def main():
    """
    Main function to run the ADE validation process.
    """
    print("Loading data...")
    n2c2_data, sider_data = load_data()
    
    print("Checking ADE consistency...")
    validated_data = check_ade_consistency(n2c2_data, sider_data)
    
    print("Generating validation report...")
    stats = generate_validation_report(validated_data)
    
    print("\nValidation Results:")
    print(f"Total ADEs analyzed: {stats['total_ades']}")
    print(f"ADEs consistent with SIDER: {stats['consistent_ades']}")
    print(f"Consistency rate: {stats['consistency_rate']:.2f}%")
    print(f"SIDER coverage rate: {stats['coverage_rate']:.2f}%")
    print("\nDetailed results have been saved to 'n2c2_with_sider_context.csv'")
    print("Drug-specific statistics have been saved to 'drug_validation_stats.csv'")

if __name__ == "__main__":
    main()