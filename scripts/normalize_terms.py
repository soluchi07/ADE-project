#!/usr/bin/env python3
"""Normalize n2c2 ADE-Drug relations and SIDER reference files.

Outputs:
 - data/n2c2/processed/n2c2_normalized.csv
 - data/sider/processed/sider_clean.csv

Usage: python scripts/normalize_terms.py
"""
import os
import re
import csv
import pandas as pd
from pathlib import Path

try:
    from rapidfuzz import process, fuzz
    _backend = 'rapidfuzz'
except Exception:
    try:
        from fuzzywuzzy import process, fuzz
        _backend = 'fuzzywuzzy'
    except Exception:
        process = None
        fuzz = None
        _backend = 'none'


def normalize_text(s: str) -> str:
    """Normalize text: lowercase, remove punctuation, strip whitespace."""
    if pd.isna(s):
        return ''
    s = str(s)
    s = s.lower()
    # Remove punctuation but keep hyphens and slashes
    s = re.sub(r"[\"\'\.,;:\!\?\(\)\[\]\{\}]", "", s)
    s = s.strip()
    return s


def load_drug_names(path: Path) -> dict:
    """Load drug_names.tsv and return dict of normalized drug names.
    
    Format: STITCH_ID | drug_name (NOTE: first column is STITCH flat ID, not ATC)
    Returns: {normalized_name: original_name}
    """
    if not path.exists():
        return {}
    
    df = pd.read_csv(path, sep='\t', dtype=str, engine='python', header=None)
    
    if len(df.columns) < 2:
        print(f'   WARNING: {path.name} has unexpected format')
        return {}
    
    df.columns = ['stitch_id', 'drug_name'] + [f'col_{i}' for i in range(2, len(df.columns))]
    
    drug_dict = {}
    for _, row in df.iterrows():
        drug_name = row['drug_name']
        if pd.notna(drug_name) and str(drug_name).strip():
            normalized = normalize_text(drug_name)
            if normalized and normalized not in drug_dict:
                drug_dict[normalized] = str(drug_name).strip()
    
    return drug_dict


def load_meddra_side_effects(path: Path) -> dict:
    """Load meddra_all_se.tsv and return dict of side effect names.
    
    Format: STITCH_flat | STITCH_stereo | UMLS_label | MedDRA_type | UMLS_MedDRA | side_effect_name
    Returns: {normalized_name: original_name}
    """
    if not path.exists():
        return {}
    
    df = pd.read_csv(path, sep='\t', dtype=str, engine='python', header=None, compression='infer')
    
    if len(df.columns) < 6:
        print(f'   WARNING: {path.name} has unexpected format (expected 6 columns)')
        return {}
    
    df.columns = ['stitch_flat', 'stitch_stereo', 'umls_label', 'meddra_type', 
                  'umls_meddra', 'side_effect'] + [f'col_{i}' for i in range(6, len(df.columns))]
    
    se_dict = {}
    for _, row in df.iterrows():
        se_name = row['side_effect']
        if pd.notna(se_name) and str(se_name).strip():
            normalized = normalize_text(se_name)
            if normalized and normalized not in se_dict:
                se_dict[normalized] = str(se_name).strip()
    
    return se_dict


def build_sider_clean(drug_names_path: Path, meddra_path: Path, 
                      atc_stitch_path: Path = None) -> pd.DataFrame: # type: ignore
    """Build SIDER drug-side effect pairs.
    
    The ATC code in drug_names.tsv matches the STITCH flat ID in meddra_all_se.tsv.
    This allows direct joining.
    
    Returns DataFrame with columns: drug_name, side_effect
    """
    if not drug_names_path.exists() or not meddra_path.exists():
        return pd.DataFrame(columns=['drug_name', 'side_effect'])
    
    # Load drug names: ATC_code (STITCH flat ID) -> drug_name
    drug_df = pd.read_csv(drug_names_path, sep='\t', dtype=str, 
                          engine='python', header=None)
    
    if len(drug_df.columns) < 2:
        print(f'   WARNING: {drug_names_path.name} has unexpected format')
        return pd.DataFrame(columns=['drug_name', 'side_effect'])
    
    drug_df.columns = ['stitch_id', 'drug_name'] + [f'col_{i}' for i in range(2, len(drug_df.columns))]
    
    # Create STITCH ID -> drug name mapping
    stitch_to_drug = {}
    for _, row in drug_df.iterrows():
        stitch_id = str(row['stitch_id']).strip()
        drug_name = row['drug_name']
        if pd.notna(drug_name) and str(drug_name).strip():
            stitch_to_drug[stitch_id] = str(drug_name).strip()
    
    # Load meddra: STITCH flat ID -> side effects
    meddra_df = pd.read_csv(meddra_path, sep='\t', dtype=str, 
                            engine='python', header=None, compression='infer')
    
    if len(meddra_df.columns) < 6:
        print(f'   WARNING: {meddra_path.name} has unexpected format (expected 6 columns)')
        return pd.DataFrame(columns=['drug_name', 'side_effect'])
    
    meddra_df.columns = ['stitch_flat', 'stitch_stereo', 'umls_label', 'meddra_type',
                         'umls_meddra', 'side_effect'] + [f'col_{i}' for i in range(6, len(meddra_df.columns))]
    
    # Build drug-side effect pairs by joining on STITCH ID
    pairs = []
    for _, row in meddra_df.iterrows():
        stitch_flat = str(row['stitch_flat']).strip()
        se = row['side_effect']
        
        if pd.isna(se) or not str(se).strip():
            continue
        
        # Look up drug name using STITCH flat ID
        drug_name = stitch_to_drug.get(stitch_flat)
        
        if drug_name:
            drug_norm = normalize_text(drug_name)
            se_norm = normalize_text(se)
            
            if drug_norm and se_norm:
                pairs.append({
                    'drug_name': drug_norm,
                    'side_effect': se_norm
                })
    
    df_pairs = pd.DataFrame(pairs)
    if len(df_pairs) > 0:
        df_pairs = df_pairs.drop_duplicates()
    
    print(f'   Successfully joined {len(stitch_to_drug)} drugs with side effects')
    
    return df_pairs


def build_sider_with_mapping(drug_names_path: Path, meddra_path: Path, 
                             atc_stitch_path: Path) -> pd.DataFrame:
    """Build actual SIDER drug-SE pairs using ATC-STITCH mapping."""
    
    # Load ATC to STITCH mapping
    atc_stitch_df = pd.read_csv(atc_stitch_path, sep='\t', dtype=str, 
                                 engine='python', header=None)
    atc_stitch_df.columns = ['stitch', 'atc'] + [f'col_{i}' for i in range(2, len(atc_stitch_df.columns))]
    
    # Load drug names (ATC -> drug name)
    drug_df = pd.read_csv(drug_names_path, sep='\t', dtype=str, 
                          engine='python', header=None)
    drug_df.columns = ['atc', 'drug_name'] + [f'col_{i}' for i in range(2, len(drug_df.columns))]
    
    # Load meddra (STITCH -> side effect)
    meddra_df = pd.read_csv(meddra_path, sep='\t', dtype=str, 
                            engine='python', header=None, compression='infer')
    meddra_df.columns = ['stitch_flat', 'stitch_stereo', 'umls_label', 'meddra_type',
                         'umls_meddra', 'side_effect'] + [f'col_{i}' for i in range(6, len(meddra_df.columns))]
    
    # Create mappings
    atc_to_drug = dict(zip(drug_df['atc'], drug_df['drug_name']))
    stitch_to_atc = dict(zip(atc_stitch_df['stitch'], atc_stitch_df['atc']))
    
    # Build pairs
    pairs = []
    for _, row in meddra_df.iterrows():
        stitch_flat = str(row['stitch_flat'])
        se = row['side_effect']
        
        if pd.isna(se) or not str(se).strip():
            continue
        
        # Map STITCH -> ATC -> drug name
        atc = stitch_to_atc.get(stitch_flat)
        if atc:
            drug_name = atc_to_drug.get(atc)
            if drug_name and str(drug_name).strip():
                drug_norm = normalize_text(drug_name)
                se_norm = normalize_text(se)
                if drug_norm and se_norm:
                    pairs.append({
                        'drug_name': drug_norm,
                        'side_effect': se_norm
                    })
    
    return pd.DataFrame(pairs).drop_duplicates()


def fuzzy_match(term: str, reference_dict: dict, threshold=85):
    """Fuzzy match a term against a reference dictionary.
    
    Args:
        term: normalized term to match
        reference_dict: {normalized: original} mapping
        threshold: minimum match score
    
    Returns: (matched_normalized, matched_original, score)
    """
    if not term or not reference_dict:
        return '', '', 0
    
    choices = list(reference_dict.keys())
    
    if _backend == 'rapidfuzz':
        res = process.extractOne(term, choices, scorer=fuzz.WRatio) # type: ignore
        if not res:
            return '', '', 0
        match_norm, score, _idx = res # type: ignore
        if score >= threshold:
            return match_norm, reference_dict[match_norm], int(score)
        return '', '', int(score)
    
    elif _backend == 'fuzzywuzzy':
        res = process.extractOne(term, choices, scorer=fuzz.WRatio) # type: ignore
        if not res:
            return '', '', 0
        match_norm, score = res # type: ignore
        if score >= threshold:
            return match_norm, reference_dict[match_norm], int(score)
        return '', '', int(score)
    
    else:
        # Fallback: exact match
        if term in reference_dict:
            return term, reference_dict[term], 100
        for choice in choices:
            if term in choice or choice in term:
                return choice, reference_dict[choice], 80
        return '', '', 0


def main():
    base = Path(__file__).resolve().parents[1]
    n2c2_in = base / 'data' / 'n2c2' / 'processed' / 'ade_drug_relations.csv'
    
    # Allow alternative filename
    alt = base / 'data' / 'n2c2' / 'processed' / 'drug_relations.csv'
    if not n2c2_in.exists() and alt.exists():
        n2c2_in = alt

    drug_names_path = base / 'data' / 'sider' / 'raw' / 'drug_names.tsv'
    meddra_path = base / 'data' / 'sider' / 'raw' / 'meddra_all_se.tsv'
    atc_stitch_path = base / 'data' / 'sider' / 'raw' / 'drug_atc.tsv'

    print(f'Normalization script: backend = {_backend}')
    print('='*60)

    # ========== STEP 1: Load reference dictionaries ==========
    print('\n[1] Loading SIDER reference dictionaries...')
    drug_dict = load_drug_names(drug_names_path)
    se_dict = load_meddra_side_effects(meddra_path)
    
    print(f'   Loaded {len(drug_dict)} unique drug names')
    print(f'   Loaded {len(se_dict)} unique side effect names')

    # ========== STEP 2: Build SIDER clean dataset ==========
    print('\n[2] Building SIDER clean dataset...')
    sider_pairs = build_sider_clean(drug_names_path, meddra_path, atc_stitch_path)
    
    sider_out_dir = base / 'data' / 'sider' / 'processed'
    sider_out_dir.mkdir(parents=True, exist_ok=True)
    sider_out = sider_out_dir / 'sider_clean.csv'
    
    sider_pairs.to_csv(sider_out, index=False)
    
    valid_pairs = sider_pairs[(sider_pairs['drug_name'] != '') & (sider_pairs['side_effect'] != '')]
    print(f'   Wrote {len(valid_pairs)} drug-side effect pairs to {sider_out}')
    
    if len(valid_pairs) == 0:
        print(f'   WARNING: No valid pairs found. Check that STITCH IDs match between files.')

    # ========== STEP 3: Normalize n2c2 data ==========
    print('\n[3] Normalizing n2c2 ADE-Drug relations...')
    
    if not n2c2_in.exists():
        print(f'   ERROR: n2c2 relations file not found at {n2c2_in}')
        return

    df = pd.read_csv(n2c2_in, dtype=str)
    print(f'   Loaded {len(df)} relations from {n2c2_in.name}')

    # Identify drug and ADE columns
    cols = list(df.columns)
    drug_col = None
    ade_col = None
    
    for c in cols:
        c_lower = c.lower()
        if 'drug' in c_lower and 'ade' not in c_lower:
            drug_col = c
        if 'ade' in c_lower or 'side' in c_lower or 'effect' in c_lower:
            ade_col = c
    
    if not drug_col or not ade_col:
        # Fallback to first two string columns
        drug_col = cols[0]
        ade_col = cols[1] if len(cols) > 1 else cols[0]
    
    print(f'   Using columns: drug="{drug_col}", ade="{ade_col}"')

    # Normalize and match
    norm_drugs = []
    norm_ades = []
    matched_drugs = []
    matched_drug_originals = []
    matched_ades = []
    matched_ade_originals = []
    drug_scores = []
    ade_scores = []

    for _, row in df.iterrows():
        raw_drug = row.get(drug_col, '')
        raw_ade = row.get(ade_col, '')
        
        # Normalize
        nd = normalize_text(raw_drug)
        na = normalize_text(raw_ade)
        
        # Match drugs against drug dictionary ONLY
        match_d_norm, match_d_orig, ds = fuzzy_match(nd, drug_dict, threshold=85)
        
        # Match ADEs against side effect dictionary ONLY
        match_a_norm, match_a_orig, ascore = fuzzy_match(na, se_dict, threshold=85)
        
        norm_drugs.append(nd)
        norm_ades.append(na)
        matched_drugs.append(match_d_norm)
        matched_drug_originals.append(match_d_orig)
        matched_ades.append(match_a_norm)
        matched_ade_originals.append(match_a_orig)
        drug_scores.append(ds)
        ade_scores.append(ascore)

    df['drug_normalized'] = norm_drugs
    df['ade_normalized'] = norm_ades
    df['drug_matched'] = matched_drugs
    df['drug_matched_original'] = matched_drug_originals
    df['ade_matched'] = matched_ades
    df['ade_matched_original'] = matched_ade_originals
    df['drug_match_score'] = drug_scores
    df['ade_match_score'] = ade_scores

    # Output
    n2c2_out_dir = base / 'data' / 'n2c2' / 'processed'
    n2c2_out_dir.mkdir(parents=True, exist_ok=True)
    n2c2_out = n2c2_out_dir / 'n2c2_normalized.csv'
    
    df.to_csv(n2c2_out, index=False)
    print(f'   Wrote {n2c2_out}')

    # ========== STEP 4: Summary ==========
    print('\n[4] Summary:')
    print('='*60)
    print(f'Total n2c2 relations: {len(df)}')
    
    high_drug_matches = sum(1 for v in df['drug_match_score'] if int(v) >= 85)
    high_ade_matches = sum(1 for v in df['ade_match_score'] if int(v) >= 85)
    
    print(f'Drugs matched (score ≥85): {high_drug_matches} ({high_drug_matches/len(df)*100:.1f}%)')
    print(f'ADEs matched (score ≥85): {high_ade_matches} ({high_ade_matches/len(df)*100:.1f}%)')
    
    both_matched = sum(1 for i in range(len(df)) 
                      if int(df.iloc[i]['drug_match_score']) >= 85 
                      and int(df.iloc[i]['ade_match_score']) >= 85)
    print(f'Both matched (score ≥85): {both_matched} ({both_matched/len(df)*100:.1f}%)')
    print('='*60)


if __name__ == '__main__':
    main()