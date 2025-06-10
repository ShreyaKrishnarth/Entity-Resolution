#!/usr/bin/env python3

import pandas as pd
import sys
from data_processor import DataProcessor
from deduplication_engine import DeduplicationEngine

def test_deduplication():
    """Test deduplication with actual data files."""
    
    # Load the actual datasets
    try:
        df1 = pd.read_csv('attached_assets/bd df 1000_1749585466102.csv')
        df2 = pd.read_csv('attached_assets/ts df 1000_1749585466103.csv')
        
        print(f"Dataset 1 shape: {df1.shape}")
        print(f"Dataset 2 shape: {df2.shape}")
        
        # Create datasets dictionary for processing
        datasets = {
            'dataset1': {'name': 'bd_df_1000.csv', 'data': df1},
            'dataset2': {'name': 'ts_df_1000.csv', 'data': df2}
        }
        
        # Process data
        processor = DataProcessor()
        processed_data = processor.process_datasets(datasets)
        
        print(f"Combined data shape: {processed_data['combined_data'].shape}")
        print(f"Common columns: {processed_data['common_columns']}")
        
        # Test deduplication with lower threshold
        dedup_engine = DeduplicationEngine()
        
        config = {
            'similarity_threshold': 0.4,  # Even lower threshold for testing
            'algorithm': 'fuzzy_partial',  # More lenient algorithm
            'primary_field': 'name',
            'use_secondary_fields': False  # Disable secondary for speed
        }
        
        print("\nTesting deduplication...")
        duplicates = dedup_engine.find_duplicates(
            processed_data['combined_data'], 
            config
        )
        
        print(f"Found {len(duplicates)} duplicate pairs")
        
        # Show first few duplicates
        for i, dup in enumerate(duplicates[:5]):
            print(f"\nDuplicate {i+1} (similarity: {dup['similarity']:.3f}):")
            print(f"  Product 1: {dup['product1'].get('name', 'N/A')}")
            print(f"  Product 2: {dup['product2'].get('name', 'N/A')}")
            print(f"  Sources: {dup['product1'].get('source', 'N/A')} vs {dup['product2'].get('source', 'N/A')}")
        
        if len(duplicates) > 0:
            # Test master catalogue creation
            print("\nCreating master catalogue...")
            master_catalogue = dedup_engine.create_master_catalogue(
                processed_data['combined_data'], 
                duplicates
            )
            print(f"Master catalogue shape: {master_catalogue.shape}")
            print(f"Master catalogue columns: {list(master_catalogue.columns)}")
            
            return True
        else:
            print("ERROR: No duplicates found - this indicates a bug in the deduplication logic")
            return False
            
    except Exception as e:
        print(f"Error during testing: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_deduplication()
    sys.exit(0 if success else 1)