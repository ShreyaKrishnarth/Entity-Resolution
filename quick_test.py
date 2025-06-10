#!/usr/bin/env python3

import pandas as pd
import sys
from rapidfuzz import fuzz

def quick_duplicate_test():
    """Quick test to verify duplicate detection works with actual data."""
    
    # Load datasets
    df1 = pd.read_csv('attached_assets/bd df 1000_1749585466102.csv')
    df2 = pd.read_csv('attached_assets/ts df 1000_1749585466103.csv')
    
    # Extract product names for comparison
    names1 = df1['product_name'].dropna().str.lower().str.strip().tolist()
    names2 = df2['name'].dropna().str.lower().str.strip().tolist()
    
    print(f"Dataset 1: {len(names1)} products")
    print(f"Dataset 2: {len(names2)} products")
    
    # Find potential matches manually
    matches = []
    threshold = 0.8
    
    print("\nSearching for duplicates...")
    
    # Test a small subset for verification
    for i, name1 in enumerate(names1[:20]):  # Test first 20 from dataset 1
        for j, name2 in enumerate(names2):
            if name1 and name2:
                similarity = fuzz.token_sort_ratio(name1, name2) / 100.0
                if similarity >= threshold:
                    matches.append({
                        'name1': name1,
                        'name2': name2,
                        'similarity': similarity,
                        'idx1': i,
                        'idx2': j
                    })
    
    print(f"Found {len(matches)} potential matches in subset")
    
    # Show matches
    for match in matches[:10]:
        print(f"  {match['similarity']:.3f}: '{match['name1']}' vs '{match['name2']}'")
    
    # Test specific known overlaps
    print("\nTesting specific known products:")
    test_cases = [
        ("amazon", "amazon"),
        ("adobe", "adobe"),
        ("microsoft", "microsoft"),
        ("google", "google")
    ]
    
    for term1, term2 in test_cases:
        matches_term = []
        for name1 in names1:
            if term1.lower() in name1.lower():
                for name2 in names2:
                    if term2.lower() in name2.lower():
                        sim = fuzz.partial_ratio(name1, name2) / 100.0
                        if sim >= 0.4:
                            matches_term.append((name1, name2, sim))
        
        if matches_term:
            print(f"  {term1.upper()} matches: {len(matches_term)}")
            for match in matches_term[:3]:
                print(f"    {match[2]:.3f}: '{match[0]}' vs '{match[1]}'")
    
    return len(matches) > 0

if __name__ == "__main__":
    success = quick_duplicate_test()
    sys.exit(0 if success else 1)