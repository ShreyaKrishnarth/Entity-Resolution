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
    
    # Extract the specific columns for comparison with aggressive normalization
    bd_names = bd_df['product_name'].dropna().astype(str).str.lower().str.strip().str.replace(r'\s+', ' ', regex=True).tolist()
    ts_names = ts_df['name'].dropna().astype(str).str.lower().str.strip().str.replace(r'\s+', ' ', regex=True).tolist()
    
    print(f"BD product names: {len(bd_names)}")
    print(f"TS product names: {len(ts_names)}")
    
    # Sample first few names to verify data
    print("\nSample BD product names:")
    for i, name in enumerate(bd_names[:5]):
        print(f"  {i+1}. {name}")
    
    print("\nSample TS product names:")
    for i, name in enumerate(ts_names[:5]):
        print(f"  {i+1}. {name}")
    
    # Find duplicates using multiple fuzzy matching algorithms
    duplicates = []
    threshold = 0.75  # Balanced threshold for quality matches
    
    print(f"\nSearching for duplicates with threshold {threshold}...")
    
    for i, bd_name in enumerate(bd_names):
        if not bd_name.strip():
            continue
            
        for j, ts_name in enumerate(ts_names):
            if not ts_name.strip():
                continue
            
            # Use multiple algorithms for better matching
            token_sort_score = fuzz.token_sort_ratio(bd_name, ts_name) / 100.0
            token_set_score = fuzz.token_set_ratio(bd_name, ts_name) / 100.0
            ratio_score = fuzz.ratio(bd_name, ts_name) / 100.0
            
            # Check for exact match after normalization (case-insensitive)
            if bd_name == ts_name:
                similarity = 1.0
            # For very short names, require higher similarity to avoid false positives
            elif len(bd_name) <= 10 or len(ts_name) <= 10:
                similarity = max(token_sort_score, ratio_score)  # Don't use token_set for short names
            else:
                # For longer names, use the best score from all algorithms
                similarity = max(token_sort_score, token_set_score, ratio_score)
            
            if similarity >= threshold:
                duplicates.append({
                    'bd_idx': i,
                    'ts_idx': j,
                    'bd_name': bd_name,
                    'ts_name': ts_name,
                    'similarity': similarity,
                    'token_sort': token_sort_score,
                    'token_set': token_set_score,
                    'ratio': ratio_score
                })
    
    print(f"Found {len(duplicates)} potential duplicates")
    
    # Show top matches
    duplicates.sort(key=lambda x: x['similarity'], reverse=True)
    
    print("\nTop duplicate matches:")
    for i, dup in enumerate(duplicates[:15]):
        print(f"  {i+1:2d}. {dup['similarity']:.3f}: '{dup['bd_name']}' vs '{dup['ts_name']}'")
    
    # Create master catalogue with mixed order (not sorted)
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
                'source': 'bd + ts',
                'original_order': min(bd_idx, ts_idx + 1000)  # Maintain relative order
            }
            
            master_records.append(merged_record)
            used_bd_indices.add(bd_idx)
            used_ts_indices.add(ts_idx)
    
    # Add unique records from BD dataset with their original positions
    for i, row in bd_df.iterrows():
        if i not in used_bd_indices:
            master_records.append({
                'product_name': row['product_name'],
                'description': row['description'] if pd.notna(row['description']) else '',
                'url': row['seller_website'] if pd.notna(row['seller_website']) else '',
                'category': row['main_category'] if pd.notna(row['main_category']) else '',
                'source': 'bd',
                'original_order': i
            })
    
    # Add unique records from TS dataset with their original positions
    for i, row in ts_df.iterrows():
        if i not in used_ts_indices:
            master_records.append({
                'product_name': row['name'],
                'description': row['description'] if pd.notna(row['description']) else '',
                'url': row['url'] if pd.notna(row['url']) else '',
                'category': row['category'] if pd.notna(row['category']) else '',
                'source': 'ts',
                'original_order': i + 1000  # Offset to maintain mixed order
            })
    
    # Sort by original order to maintain natural jumbled sequence
    master_records.sort(key=lambda x: x['original_order'])
    
    # Remove the ordering column
    for record in master_records:
        del record['original_order']
    
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