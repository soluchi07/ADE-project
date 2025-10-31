import os, re, pandas as pd
import glob

def parse_ann_file(ann_path):
    entities, relations = [], []
    with open(ann_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.startswith('T'):
                tid, label_text = line.strip().split('\t', 1)
                label, start, end, text = re.split(r'\s+', label_text, maxsplit=3)
                entities.append((tid, label, text))
            elif line.startswith('R'):
                rid, rest = line.strip().split('\t')
                rel_type, args = rest.split(' ', 1)

                arg1_match = re.search(r'Arg1:(T\d+)', args)
                arg2_match = re.search(r'Arg2:(T\d+)', args)

                if arg1_match and arg2_match:
                    arg1 = arg1_match.group(1)
                    arg2 = arg2_match.group(1)
                    relations.append((rid, rel_type, arg1, arg2))
                else:
                    # Skip this line or log the error
                    print(f"Skipping malformed relation: {line.strip()}")
                
    return pd.DataFrame(entities, columns=["entity_id", "label", "text"]), \
           pd.DataFrame(relations, columns=["rel_id", "relation", "arg1", "arg2"])

def map_relations():
    
    # Load the processed CSVs
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "..", "data", "n2c2", "processed")

    entities_df = pd.read_csv(os.path.join(data_dir, "n2c2_entities.csv"))
    relations_df = pd.read_csv(os.path.join(data_dir, "n2c2_relations.csv"))

    print(f"Loaded {len(entities_df)} entities and {len(relations_df)} relations")
    print(f"\nEntity labels: {entities_df['label'].value_counts()}")
    print(f"\nRelation types: {relations_df['relation'].value_counts()}")

    # Filter for ADE-Drug relations only
    relations = relations_df[relations_df['relation'] == 'ADE-Drug']

    # Merge to get the text for arg1 (should be ADE) and arg2 (should be Drug)
    merged = relations.merge(
        entities_df, 
        left_on='arg1', 
        right_on='entity_id'
    ).merge(
        entities_df, 
        left_on='arg2', 
        right_on='entity_id', 
        suffixes=('_ade', '_drug')
    )

    # Select relevant columns for the final output
    output = merged[['text_ade', 'text_drug', 'source_file']]
    # output = output.rename(columns={'source_file_ade': 'source_file'})

    # Save the mapped ADE-Drug pairs
    output.to_csv(os.path.join(data_dir, "ade_drug_relations.csv"), index=False)

    print(f"\nCreated ade_drug_relations.csv with {len(output)} ADE-Drug pairs")
    print(f"\nSample pairs:")
    print(output.head(10))

    # Optional: Verify the mapping makes sense
    print(f"\nLabel verification:")
    print(f"arg1 (ADE) labels: {merged['label_ade'].value_counts()}")
    print(f"arg2 (Drug) labels: {merged['label_drug'].value_counts()}")

def n2c2_extract_all():
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Using test directory but can change if needed (Task2 has no relationships)
    ann_files = glob.glob(os.path.join(script_dir, "..", "data", "n2c2", "raw", "test", "*.ann"))
    
    output_dir = os.path.join(script_dir, "..", "data", "n2c2", "processed")
    
    print(f"Found {len(ann_files)} files")
    all_entities = []
    all_relations = []
    
    for ann_file in ann_files:
        print(f"Processing {ann_file}...")
        entities_df, relations_df = parse_ann_file(ann_file)
        
        base_name = os.path.splitext(os.path.basename(ann_file))[0]

        # Make IDs globally unique by prefixing with filename
        entities_df['entity_id'] = base_name + '_' + entities_df['entity_id']
        relations_df['rel_id'] = base_name + '_' + relations_df['rel_id']
        relations_df['arg1'] = base_name + '_' + relations_df['arg1']
        relations_df['arg2'] = base_name + '_' + relations_df['arg2']

        entities_df['source_file'] = base_name
        relations_df['source_file'] = base_name

        all_entities.append(entities_df)
        all_relations.append(relations_df)

        # Concatenate all dataframes
        combined_entities = pd.concat(all_entities, ignore_index=True)
        combined_relations = pd.concat(all_relations, ignore_index=True)

        filtered_entities = combined_entities[combined_entities['label'].isin(['Drug', 'ADE'])]
    
        # Filter relations: keep only ADE-Drug
        filtered_relations = combined_relations[combined_relations['relation'] == 'ADE-Drug']

        filtered_entities.to_csv(os.path.join(output_dir, "n2c2_entities.csv"), index=False)
        filtered_relations.to_csv(os.path.join(output_dir, "n2c2_relations.csv"), index=False)
        
    
    print(f"Done! Processed {len(ann_files)} files")
    print(f"Total entities (Drug/ADE): {len(filtered_entities)}")
    print(f"Total ADE-Drug relations: {len(filtered_relations)}")





if __name__ == "__main__":
    
    n2c2_extract_all()
    map_relations()