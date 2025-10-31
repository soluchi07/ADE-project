#!/usr/bin/env python3
"""Filter and reduce noise in ADE detection results.

This script implements the noise reduction and filtering step described in the project overview.
It uses SIDER frequency data and n2c2 validation results to:
1. Remove low-confidence drug-ADE pairs
2. Keep high-frequency SIDER-validated pairs
3. Preserve novel ADEs when multiple strong signals exist
4. Generate reports on filtered results

Usage: python scripts/filter_ades.py [--threshold FLOAT] [--min-freq INT]
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict


def load_data(base_path: Path):
    """Load the validated n2c2 data and SIDER reference data."""
    n2c2_path = base_path / 'data' / 'n2c2' / 'processed'
    sider_path = base_path / 'data' / 'sider' / 'processed'
    
    # Load SIDER frequency data
    sider_df = pd.read_csv(sider_path / 'sider_clean.csv')
    print(f"Loaded {len(sider_df)} SIDER drug-ADE pairs")
    
    # Load n2c2 validation results
    n2c2_df = pd.read_csv(n2c2_path / 'n2c2_with_sider_context.csv')
    print(f"Loaded {len(n2c2_df)} n2c2 drug-ADE pairs")
    
    return sider_df, n2c2_df


def analyze_drug_patterns(n2c2_df: pd.DataFrame) -> dict:
    """Analyze drug mention patterns to identify consistent ADEs.
    
    Returns a dict with drug-level statistics to help identify reliable patterns.
    """
    drug_patterns = defaultdict(lambda: {
        'total_mentions': 0,
        'ades': defaultdict(int),
        'consistent_ades': set(),
        'sider_validated': 0
    })
    
    for _, row in n2c2_df.iterrows():
        drug = row['drug_norm']
        ade = row['ade_norm']
        
        stats = drug_patterns[drug]
        stats['total_mentions'] += 1 # type: ignore
        stats['ades'][ade] += 1 # type: ignore
        
        if row['is_consistent']:
            stats['consistent_ades'].add(ade) # type: ignore
            stats['sider_validated'] += 1 # type: ignore
    
    # Convert to regular dict for return
    return {k: dict(v) for k, v in drug_patterns.items()}


def should_keep_ade(
    row: pd.Series,
    drug_patterns: dict,
    sider_freqs: dict,
    min_freq: int = 2,
    consistency_threshold: float = 0.4
) -> tuple[bool, str]:
    """Determine if an ADE mention should be kept based on multiple criteria.
    
    Returns:
    - bool: Whether to keep the ADE
    - str: Reason for the decision
    """
    drug = row['drug_norm']
    ade = row['ade_norm']
    
    # Get drug statistics
    drug_stats = drug_patterns.get(drug, {})
    total_mentions = drug_stats.get('total_mentions', 0)
    ade_mentions = drug_stats.get('ades', {}).get(ade, 0)
    sider_validated = drug_stats.get('sider_validated', 0)
    
    # Calculate confidence metrics
    mention_ratio = ade_mentions / total_mentions if total_mentions > 0 else 0
    sider_consistency = sider_validated / total_mentions if total_mentions > 0 else 0
    
    # Check SIDER frequency
    sider_freq = sider_freqs.get((drug, ade), 0)
    
    # Decision logic
    if row['is_consistent'] and sider_freq >= min_freq:
        return True, "high_confidence_sider"
    
    if mention_ratio >= 0.5 and ade_mentions >= 2:
        return True, "strong_local_signal"
        
    if sider_consistency >= consistency_threshold and ade_mentions >= 2:
        return True, "consistent_drug_pattern"
    
    if total_mentions >= 5 and mention_ratio >= 0.3:
        return True, "frequent_association"
    
    return False, "insufficient_evidence"


def filter_ades(
    n2c2_df: pd.DataFrame,
    sider_df: pd.DataFrame,
    min_freq: int = 2,
    consistency_threshold: float = 0.4
) -> pd.DataFrame:
    """Filter ADEs based on SIDER frequencies and local patterns."""
    
    # Build SIDER frequency lookup
    sider_freqs = {
        (row['drug_norm'], row['ade_norm']): row['frequency']
        for _, row in sider_df.iterrows()
    }
    
    # Analyze drug mention patterns
    drug_patterns = analyze_drug_patterns(n2c2_df)
    
    # Apply filtering
    keep_rows = []
    reasons = []
    
    for _, row in n2c2_df.iterrows():
        keep, reason = should_keep_ade(
            row, 
            drug_patterns,
            sider_freqs,
            min_freq,
            consistency_threshold
        )
        keep_rows.append(keep)
        reasons.append(reason)
    
    # Add decision columns
    n2c2_df['kept'] = keep_rows
    n2c2_df['filter_reason'] = reasons
    
    return n2c2_df


def write_reports(filtered_df: pd.DataFrame, output_dir: Path):
    """Generate detailed reports on the filtering results."""
    
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Write main filtered results
    filtered_df.to_csv(output_dir / 'ade_filtered_results.csv', index=False)
    
    # Generate summary by reason
    reason_summary = (
        filtered_df
        .groupby('filter_reason')
        .agg({
            'drug_norm': 'count',
            'kept': 'sum'
        })
        .round(2)
    )
    reason_summary.to_csv(output_dir / 'filter_reason_summary.csv')
    
    # Generate drug-level summary
    drug_summary = (
        filtered_df
        .groupby('drug_norm')
        .agg({
            'ade_norm': 'count',
            'is_consistent': 'sum',
            'kept': 'sum'
        })
        .round(2)
    )
    drug_summary.to_csv(output_dir / 'drug_summary.csv')
    
    # Print summary statistics
    total = len(filtered_df)
    kept = filtered_df['kept'].sum()
    sider_validated = filtered_df['is_consistent'].sum()
    
    print("\nFiltering Results:")
    print(f"Total ADEs analyzed: {total}")
    print(f"ADEs kept: {kept} ({kept/total*100:.1f}%)")
    print(f"SIDER validated: {sider_validated} ({sider_validated/total*100:.1f}%)")
    print("\nDetailed reports written to:")
    print(f"- {output_dir / 'ade_filtered_results.csv'}")
    print(f"- {output_dir / 'filter_reason_summary.csv'}")
    print(f"- {output_dir / 'drug_summary.csv'}")


def main():
    parser = argparse.ArgumentParser(description="Filter and reduce noise in ADE detection results")
    parser.add_argument('--threshold', type=float, default=0.4,
                      help="Consistency threshold for keeping ADEs (default: 0.4)")
    parser.add_argument('--min-freq', type=int, default=2,
                      help="Minimum SIDER frequency to consider high-confidence (default: 2)")
    args = parser.parse_args()
    
    base_path = Path(__file__).resolve().parents[1]
    
    print("Loading data...")
    sider_df, n2c2_df = load_data(base_path)
    
    print("\nApplying filters...")
    filtered_df = filter_ades(
        n2c2_df,
        sider_df,
        min_freq=args.min_freq,
        consistency_threshold=args.threshold
    )
    
    # Write reports
    output_dir = base_path / 'data' / 'n2c2' / 'processed'
    write_reports(filtered_df, output_dir)


if __name__ == '__main__':
    main()