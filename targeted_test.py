#!/usr/bin/env python3

import pandas as pd
import sys
from rapidfuzz import fuzz

def test_specific_columns():
    """Test deduplication using specific columns: product_name from bd and name from ts."""
    
    # Load the new datasets
    bd_df = pd.read_csv('attached_assets/bd df 1000_1749590902120.csv')
    ts_df = pd.read_csv('attached_assets/ts 1000 df_1749590902121.csv')
    
    print(f"BD dataset: {len(bd_df)} products")
    print(f"TS dataset: {len(ts_df)} products")
    
    # Extract the specific columns for comparison
    bd_names = bd_df['product_name'].dropna().str.lower().str.strip().tolist()
    ts_names = ts_df['name'].dropna().str.lower().str.strip().tolist()
    
    print(f"BD product names: {len(bd_names)}")
    print(f"TS product names: {len(ts_names)}")
    
    # Sample first few names to verify data
    print("\nSample BD product names:")
    for i, name in enumerate(bd_names[:5]):
        print(f"  {i+1}. {name}")
    
    print("\nSample TS product names:")
    for i, name in enumerate(ts_names[:5]):
        print(f"  {i+1}. {name}")
    
    # Find duplicates using token_sort_ratio
    duplicates = []
    threshold = 0.75
    
    print(f"\nSearching for duplicates with threshold {threshold}...")
    
    for i, bd_name in enumerate(bd_names):
        if not bd_name.strip():
            continue
            
        for j, ts_name in enumerate(ts_names):
            if not ts_name.strip():
                continue
            
            # Use token_sort_ratio for better matching
            similarity = fuzz.token_sort_ratio(bd_name, ts_name) / 100.0
            
            if similarity >= threshold:
                duplicates.append({
                    'bd_idx': i,
                    'ts_idx': j,
                    'bd_name': bd_name,
                    'ts_name': ts_name,
                    'similarity': similarity
                })
    
    print(f"Found {len(duplicates)} potential duplicates")
    
    # Show top matches
    duplicates.sort(key=lambda x: x['similarity'], reverse=True)
    
    print("\nTop duplicate matches:")
    for i, dup in enumerate(duplicates[:15]):
        print(f"  {i+1:2d}. {dup['similarity']:.3f}: '{dup['bd_name']}' vs '{dup['ts_name']}'")
    
    # Create master catalogue with specific output format
    master_records = []
    used_bd_indices = set()
    used_ts_indices = set()
    
    # Process duplicates - merge into single records
    for dup in duplicates:
        bd_idx = dup['bd_idx']
        ts_idx = dup['ts_idx']
        
        if bd_idx not in used_bd_indices and ts_idx not in used_ts_indices:
            bd_row = bd_df.iloc[bd_idx]
            ts_row = ts_df.iloc[ts_idx]
            
            # Create merged record with required 5 columns
            merged_record = {
                'product_name': bd_row['product_name'] if pd.notna(bd_row['product_name']) else ts_row['name'],
                'description': bd_row['description'] if pd.notna(bd_row['description']) and len(str(bd_row['description'])) > 50 else ts_row['description'],
                'url': bd_row['seller_website'] if pd.notna(bd_row['seller_website']) else ts_row['url'],
                'category': bd_row['main_category'] if pd.notna(bd_row['main_category']) else ts_row['category'],
                'source': 'bd + ts'
            }
            
            master_records.append(merged_record)
            used_bd_indices.add(bd_idx)
            used_ts_indices.add(ts_idx)
    
    # Add unique records from BD dataset
    for i, row in bd_df.iterrows():
        if i not in used_bd_indices:
            master_records.append({
                'product_name': row['product_name'],
                'description': row['description'] if pd.notna(row['description']) else '',
                'url': row['seller_website'] if pd.notna(row['seller_website']) else '',
                'category': row['main_category'] if pd.notna(row['main_category']) else '',
                'source': 'bd'
            })
    
    # Add unique records from TS dataset
    for i, row in ts_df.iterrows():
        if i not in used_ts_indices:
            master_records.append({
                'product_name': row['name'],
                'description': row['description'] if pd.notna(row['description']) else '',
                'url': row['url'] if pd.notna(row['url']) else '',
                'category': row['category'] if pd.notna(row['category']) else '',
                'source': 'ts'
            })
    
    # Create final master catalogue
    master_df = pd.DataFrame(master_records)
    
    print(f"\nMaster catalogue statistics:")
    print(f"Total records: {len(master_df)}")
    print(f"Original BD: {len(bd_df)}")
    print(f"Original TS: {len(ts_df)}")
    print(f"Original total: {len(bd_df) + len(ts_df)}")
    print(f"Duplicates merged: {len(duplicates)}")
    print(f"Space saved: {len(bd_df) + len(ts_df) - len(master_df)} records")
    
    # Show sample of final output
    print(f"\nSample master catalogue (first 5 records):")
    sample_df = master_df[['product_name', 'description', 'url', 'category', 'source']].head()
    for i, row in sample_df.iterrows():
        print(f"\n{i+1}. Product: {row['product_name']}")
        print(f"   Description: {str(row['description'])[:100]}...")
        print(f"   URL: {row['url']}")
        print(f"   Category: {row['category']}")
        print(f"   Source: {row['source']}")
    
    # Save master catalogue
    output_file = 'master_catalogue_final.csv'
    master_df[['product_name', 'description', 'url', 'category', 'source']].to_csv(output_file, index=False)
    print(f"\nMaster catalogue saved to: {output_file}")
    
    return len(duplicates) > 0

if __name__ == "__main__":
    success = test_specific_columns()
    sys.exit(0 if success else 1)