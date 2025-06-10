#!/usr/bin/env python3

import pandas as pd
import sys
from rapidfuzz import fuzz

def create_master_catalogue():
    """Create the master catalogue with exact output format from the image."""
    
    # Load datasets
    df1 = pd.read_csv('attached_assets/bd df 1000_1749585466102.csv')
    df2 = pd.read_csv('attached_assets/ts df 1000_1749585466103.csv')
    
    print(f"Dataset 1 (bd): {len(df1)} products")
    print(f"Dataset 2 (ts): {len(df2)} products")
    
    # Standardize column names for dataset 1
    df1_clean = df1.copy()
    df1_clean['product_name'] = df1_clean['product_name']
    df1_clean['description'] = df1_clean['description']
    df1_clean['url'] = df1_clean['seller_website']
    df1_clean['category'] = df1_clean['main_category']
    df1_clean['source'] = 'bd'
    
    # Standardize column names for dataset 2
    df2_clean = df2.copy()
    df2_clean['product_name'] = df2_clean['name']
    df2_clean['description'] = df2_clean['description']
    df2_clean['url'] = df2_clean['url']
    df2_clean['category'] = df2_clean['category']
    df2_clean['source'] = 'ts'
    
    # Select only required columns
    required_cols = ['product_name', 'description', 'url', 'category', 'source']
    df1_final = df1_clean[required_cols].copy()
    df2_final = df2_clean[required_cols].copy()
    
    # Find exact and near duplicates
    duplicates_found = []
    threshold = 0.75
    
    print("\nFinding duplicates...")
    
    # Compare products between datasets
    for i, row1 in df1_final.iterrows():
        name1 = str(row1['product_name']).lower().strip() if pd.notna(row1['product_name']) else ''
        if not name1:
            continue
            
        for j, row2 in df2_final.iterrows():
            name2 = str(row2['product_name']).lower().strip() if pd.notna(row2['product_name']) else ''
            if not name2:
                continue
            
            # Calculate similarity using token sort ratio for better matching
            similarity = fuzz.token_sort_ratio(name1, name2) / 100.0
            
            if similarity >= threshold:
                duplicates_found.append({
                    'idx1': i,
                    'idx2': j,
                    'similarity': similarity,
                    'name1': name1,
                    'name2': name2,
                    'row1': row1,
                    'row2': row2
                })
    
    print(f"Found {len(duplicates_found)} duplicate pairs")
    
    # Show top duplicates
    duplicates_found.sort(key=lambda x: x['similarity'], reverse=True)
    for dup in duplicates_found[:10]:
        print(f"  {dup['similarity']:.3f}: '{dup['name1']}' vs '{dup['name2']}'")
    
    # Create master catalogue by merging duplicates
    master_records = []
    used_indices_df1 = set()
    used_indices_df2 = set()
    
    # Process duplicates - merge into single records
    for dup in duplicates_found:
        if dup['idx1'] not in used_indices_df1 and dup['idx2'] not in used_indices_df2:
            # Merge the two records
            row1, row2 = dup['row1'], dup['row2']
            
            merged_record = {
                'product_name': row1['product_name'] if pd.notna(row1['product_name']) else row2['product_name'],
                'description': row1['description'] if pd.notna(row1['description']) and len(str(row1['description'])) > len(str(row2['description'])) else row2['description'],
                'url': row1['url'] if pd.notna(row1['url']) else row2['url'],
                'category': row1['category'] if pd.notna(row1['category']) else row2['category'],
                'source': f"{row1['source']} + {row2['source']}"
            }
            
            master_records.append(merged_record)
            used_indices_df1.add(dup['idx1'])
            used_indices_df2.add(dup['idx2'])
    
    # Add unique records from dataset 1
    for i, row in df1_final.iterrows():
        if i not in used_indices_df1:
            master_records.append(row.to_dict())
    
    # Add unique records from dataset 2
    for i, row in df2_final.iterrows():
        if i not in used_indices_df2:
            master_records.append(row.to_dict())
    
    # Create final master catalogue
    master_df = pd.DataFrame(master_records)
    
    print(f"\nMaster catalogue created with {len(master_df)} products")
    print(f"Original total: {len(df1_final) + len(df2_final)} products")
    print(f"Duplicates merged: {len(duplicates_found)} pairs")
    
    # Show sample of master catalogue
    print("\nSample master catalogue records:")
    print(master_df[['product_name', 'description', 'url', 'category', 'source']].head(10).to_string(index=False))
    
    # Save to CSV
    output_file = 'master_catalogue.csv'
    master_df[['product_name', 'description', 'url', 'category', 'source']].to_csv(output_file, index=False)
    print(f"\nMaster catalogue saved to: {output_file}")
    
    return master_df

if __name__ == "__main__":
    master_catalogue = create_master_catalogue()
    sys.exit(0)