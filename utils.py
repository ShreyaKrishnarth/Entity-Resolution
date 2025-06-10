import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import logging
from datetime import datetime
import json

class Utils:
    """
    Utility functions for the product catalogue deduplication system.
    """
    
    @staticmethod
    def setup_logging(level: str = 'INFO') -> logging.Logger:
        """
        Set up logging configuration.
        
        Args:
            level: Logging level
            
        Returns:
            Configured logger
        """
        logging.basicConfig(
            level=getattr(logging, level.upper()),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        return logging.getLogger(__name__)
    
    @staticmethod
    def validate_csv_structure(df: pd.DataFrame, required_columns: List[str] = None) -> Dict[str, Any]:
        """
        Validate CSV structure and identify potential issues.
        
        Args:
            df: Input dataframe
            required_columns: List of required column names
            
        Returns:
            Validation results dictionary
        """
        validation_results = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'suggestions': []
        }
        
        # Check for empty dataframe
        if df.empty:
            validation_results['is_valid'] = False
            validation_results['errors'].append("Dataset is empty")
            return validation_results
        
        # Check required columns
        if required_columns:
            missing_cols = set(required_columns) - set(df.columns)
            if missing_cols:
                validation_results['warnings'].append(
                    f"Missing recommended columns: {', '.join(missing_cols)}"
                )
        
        # Check for completely empty columns
        empty_cols = df.columns[df.isna().all()].tolist()
        if empty_cols:
            validation_results['warnings'].append(
                f"Completely empty columns found: {', '.join(empty_cols)}"
            )
        
        # Check for high missing value percentages
        high_missing_cols = []
        for col in df.columns:
            missing_pct = (df[col].isna().sum() / len(df)) * 100
            if missing_pct > 50:
                high_missing_cols.append(f"{col} ({missing_pct:.1f}%)")
        
        if high_missing_cols:
            validation_results['warnings'].append(
                f"Columns with high missing values: {', '.join(high_missing_cols)}"
            )
        
        # Check for potential product identifier columns
        potential_id_cols = [col for col in df.columns if 'id' in col.lower()]
        if not potential_id_cols:
            validation_results['suggestions'].append(
                "Consider adding a unique product identifier column"
            )
        
        # Check for text length in description fields
        desc_cols = [col for col in df.columns if any(
            term in col.lower() for term in ['desc', 'overview', 'summary']
        )]
        
        for col in desc_cols:
            if col in df.columns:
                avg_length = df[col].astype(str).str.len().mean()
                if avg_length < 50:
                    validation_results['warnings'].append(
                        f"Short average text length in {col}: {avg_length:.1f} characters"
                    )
        
        return validation_results
    
    @staticmethod
    def calculate_data_quality_score(df: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate overall data quality score.
        
        Args:
            df: Input dataframe
            
        Returns:
            Dictionary with quality metrics
        """
        scores = {}
        
        # Completeness score (percentage of non-null values)
        total_cells = df.shape[0] * df.shape[1]
        non_null_cells = df.count().sum()
        scores['completeness'] = (non_null_cells / total_cells) * 100 if total_cells > 0 else 0
        
        # Uniqueness score (based on duplicate rows)
        duplicate_rows = df.duplicated().sum()
        scores['uniqueness'] = ((len(df) - duplicate_rows) / len(df)) * 100 if len(df) > 0 else 0
        
        # Consistency score (based on data type consistency)
        consistency_score = 100  # Start with perfect score
        
        # Check for mixed data types in expected text columns
        text_columns = ['name', 'description', 'category']
        for col in text_columns:
            if col in df.columns:
                # Check if column has consistent string-like values
                non_string_count = df[col].apply(
                    lambda x: not isinstance(x, str) and pd.notna(x)
                ).sum()
                if non_string_count > 0:
                    consistency_score -= (non_string_count / len(df)) * 10
        
        scores['consistency'] = max(consistency_score, 0)
        
        # Overall quality score (weighted average)
        scores['overall'] = (
            scores['completeness'] * 0.4 +
            scores['uniqueness'] * 0.3 +
            scores['consistency'] * 0.3
        )
        
        return scores
    
    @staticmethod
    def generate_documentation(processed_data: Dict[str, Any], 
                             duplicates: List[Dict[str, Any]], 
                             master_catalogue: pd.DataFrame) -> str:
        """
        Generate comprehensive documentation for the deduplication process.
        
        Args:
            processed_data: Processed data information
            duplicates: List of found duplicates
            master_catalogue: Final master catalogue
            
        Returns:
            Markdown documentation string
        """
        doc_content = f"""# Product Catalogue Deduplication Report

Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Executive Summary

This report details the product catalogue deduplication process performed on {len(processed_data.get('datasets', {}))} datasets, resulting in a master catalogue of unique products.

### Key Metrics
- **Original Products**: {processed_data.get('total_products', 0)}
- **Duplicate Pairs Found**: {len(duplicates)}
- **Final Unique Products**: {len(master_catalogue)}
- **Deduplication Rate**: {((processed_data.get('total_products', 0) - len(master_catalogue)) / processed_data.get('total_products', 1) * 100):.1f}%

## Data Processing Summary

### Input Datasets
"""
        
        # Add dataset information
        for key, dataset_info in processed_data.get('datasets', {}).items():
            doc_content += f"- **{key}**: {dataset_info['name']} ({len(dataset_info['data'])} records)\n"
        
        doc_content += f"""
### Data Quality Analysis
- **Common Columns**: {len(processed_data.get('common_columns', []))}
- **Missing Values**: {len(processed_data.get('missing_values_summary', pd.DataFrame()))} columns affected
- **Data Completeness**: {Utils.calculate_data_quality_score(processed_data.get('combined_data', pd.DataFrame())).get('completeness', 0):.1f}%

## Deduplication Process

### Methodology
1. **Data Preprocessing**: Standardized column names and cleaned text fields
2. **Similarity Calculation**: Applied fuzzy matching and text similarity algorithms
3. **Threshold-based Matching**: Identified duplicates above similarity threshold
4. **Conflict Resolution**: Merged duplicate records with intelligent data prioritization

### Duplicate Analysis
"""
        
        if duplicates:
            # Analyze similarity score distribution
            similarity_scores = [dup['similarity'] for dup in duplicates]
            doc_content += f"""
- **Average Similarity Score**: {np.mean(similarity_scores):.3f}
- **Highest Similarity Score**: {max(similarity_scores):.3f}
- **Lowest Similarity Score**: {min(similarity_scores):.3f}
- **Standard Deviation**: {np.std(similarity_scores):.3f}

### Top Duplicate Pairs
"""
            # Show top 5 duplicate pairs
            for i, dup in enumerate(duplicates[:5]):
                doc_content += f"""
#### Pair {i+1} (Similarity: {dup['similarity']:.3f})
- **Product 1**: {dup['product1'].get('name', 'N/A')[:100]}...
- **Product 2**: {dup['product2'].get('name', 'N/A')[:100]}...
- **Algorithm Used**: {dup.get('algorithm', 'N/A')}
"""
        
        doc_content += f"""
## Master Catalogue Statistics

### Final Catalogue Composition
- **Total Unique Products**: {len(master_catalogue)}
- **Records with Merged Data**: {master_catalogue.get('is_merged', pd.Series()).sum() if 'is_merged' in master_catalogue.columns else 'N/A'}
- **Data Sources Represented**: {master_catalogue['source'].nunique() if 'source' in master_catalogue.columns else 'N/A'}

### Quality Improvements
- **Eliminated Redundancy**: Removed {processed_data.get('total_products', 0) - len(master_catalogue)} duplicate records
- **Enhanced Data Completeness**: Merged information from multiple sources
- **Standardized Format**: Consistent column structure and data types

## Technical Details

### Algorithms Used
- **Fuzzy String Matching**: Levenshtein distance-based similarity for name matching
- **TF-IDF Cosine Similarity**: Vector space model for description comparison
- **Token-based Matching**: Word-level comparison for robust text matching

### Validation Steps
1. Cross-source duplicate detection
2. Similarity threshold validation
3. Manual review of high-confidence matches
4. Data integrity verification

## Recommendations

### Data Quality Improvements
1. **Standardize Input Formats**: Implement consistent data collection standards
2. **Enhance Product Descriptions**: Ensure comprehensive product information
3. **Regular Deduplication**: Schedule periodic catalogue maintenance

### System Enhancements
1. **Automated Pipeline**: Set up continuous deduplication processes
2. **Feedback Loop**: Implement user feedback for algorithm improvement
3. **Performance Monitoring**: Track deduplication accuracy over time

## Limitations and Considerations

- **Similarity Thresholds**: Current thresholds may require adjustment based on domain expertise
- **Algorithm Selection**: Different product types may benefit from specialized matching algorithms
- **Manual Review**: High-stakes duplicates should undergo manual verification
- **Data Dependencies**: Quality depends on completeness of source data

## Conclusion

The deduplication process successfully identified and merged duplicate products, resulting in a cleaner and more comprehensive product catalogue. The {((processed_data.get('total_products', 0) - len(master_catalogue)) / processed_data.get('total_products', 1) * 100):.1f}% deduplication rate indicates effective duplicate detection while maintaining data integrity.

---
*Report generated by Product Catalogue Deduplication System*
"""
        
        return doc_content
    
    @staticmethod
    def export_duplicate_pairs(duplicates: List[Dict[str, Any]], 
                              filename: Optional[str] = None) -> pd.DataFrame:
        """
        Export duplicate pairs to a structured format.
        
        Args:
            duplicates: List of duplicate pairs
            filename: Optional filename for CSV export
            
        Returns:
            DataFrame with duplicate pair information
        """
        if not duplicates:
            return pd.DataFrame()
        
        export_data = []
        
        for i, dup in enumerate(duplicates):
            export_data.append({
                'pair_id': i + 1,
                'product1_name': dup['product1'].get('name', ''),
                'product1_description': dup['product1'].get('description', '')[:200] + '...' if len(dup['product1'].get('description', '')) > 200 else dup['product1'].get('description', ''),
                'product1_source': dup['product1'].get('source', ''),
                'product2_name': dup['product2'].get('name', ''),
                'product2_description': dup['product2'].get('description', '')[:200] + '...' if len(dup['product2'].get('description', '')) > 200 else dup['product2'].get('description', ''),
                'product2_source': dup['product2'].get('source', ''),
                'similarity_score': dup['similarity'],
                'algorithm_used': dup.get('algorithm', 'N/A'),
                'primary_field': dup.get('primary_field', 'N/A'),
                'confidence_level': 'High' if dup['similarity'] > 0.9 else 'Medium' if dup['similarity'] > 0.7 else 'Low'
            })
        
        df = pd.DataFrame(export_data)
        
        if filename:
            df.to_csv(filename, index=False)
        
        return df
    
    @staticmethod
    def format_similarity_score(score: float) -> str:
        """
        Format similarity score for display.
        
        Args:
            score: Similarity score between 0 and 1
            
        Returns:
            Formatted score string
        """
        percentage = score * 100
        
        if percentage >= 95:
            return f"{percentage:.1f}% (Excellent)"
        elif percentage >= 85:
            return f"{percentage:.1f}% (Very Good)"
        elif percentage >= 75:
            return f"{percentage:.1f}% (Good)"
        elif percentage >= 65:
            return f"{percentage:.1f}% (Fair)"
        else:
            return f"{percentage:.1f}% (Poor)"
    
    @staticmethod
    def get_processing_statistics(processed_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate processing statistics summary.
        
        Args:
            processed_data: Processed data dictionary
            
        Returns:
            Statistics summary
        """
        stats = {
            'total_products': processed_data.get('total_products', 0),
            'datasets_count': len(processed_data.get('datasets', {})),
            'common_columns_count': len(processed_data.get('common_columns', [])),
            'missing_values_columns': len(processed_data.get('missing_values_summary', pd.DataFrame())),
            'data_quality_score': Utils.calculate_data_quality_score(
                processed_data.get('combined_data', pd.DataFrame())
            ).get('overall', 0)
        }
        
        return stats
