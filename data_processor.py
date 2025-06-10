import pandas as pd
import numpy as np
import re
from typing import Dict, List, Any, Tuple, Optional
import logging

class DataProcessor:
    """
    Handles data ingestion, cleaning, and preprocessing for product catalogue deduplication.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def process_datasets(self, datasets: Dict[str, Dict], selected_columns: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Process multiple datasets and prepare them for deduplication.
        
        Args:
            datasets: Dictionary containing dataset information
            selected_columns: List of columns to include in processing
            
        Returns:
            Dictionary containing processed data and statistics
        """
        processed_data = {
            'datasets': datasets,
            'combined_data': None,
            'total_products': 0,
            'dataset1_count': 0,
            'dataset2_count': 0,
            'common_columns': [],
            'missing_values_summary': pd.DataFrame(),
            'column_stats': pd.DataFrame()
        }
        
        try:
            # Extract dataframes
            dfs = []
            dataset_names = []
            
            for key, dataset_info in datasets.items():
                df = dataset_info['data'].copy()
                
                # Filter columns if specified
                if selected_columns:
                    available_cols = [col for col in selected_columns if col in df.columns]
                    if available_cols:
                        df = df[available_cols].copy()
                    else:
                        self.logger.warning(f"No selected columns found in {key}")
                
                df['source'] = dataset_info['name']  # Add source tracking
                dfs.append(df)
                dataset_names.append(key)
                
                if key == 'dataset1':
                    processed_data['dataset1_count'] = len(df)
                elif key == 'dataset2':
                    processed_data['dataset2_count'] = len(df)
            
            # Standardize column names
            standardized_dfs = []
            for df in dfs:
                df_clean = self._standardize_columns(df)
                df_clean = self._clean_data(df_clean)
                standardized_dfs.append(df_clean)
            
            # Find common columns
            common_columns = self._find_common_columns(standardized_dfs)
            processed_data['common_columns'] = common_columns
            
            # Combine datasets
            combined_df = self._combine_datasets(standardized_dfs, common_columns)
            processed_data['combined_data'] = combined_df
            processed_data['total_products'] = len(combined_df)
            
            # Generate data quality statistics
            processed_data['missing_values_summary'] = self._analyze_missing_values(combined_df)
            processed_data['column_stats'] = self._generate_column_stats(combined_df)
            
            self.logger.info(f"Successfully processed {len(datasets)} datasets")
            
        except Exception as e:
            self.logger.error(f"Error processing datasets: {str(e)}")
            raise
        
        return processed_data
    
    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize column names across datasets.
        
        Args:
            df: Input dataframe
            
        Returns:
            Dataframe with standardized column names
        """
        # Column mapping for common variations
        column_mapping = {
            # Name variations
            'product_name': 'name',
            'name_clean': 'name',
            'title': 'name',
            
            # Description variations
            'description_clean': 'description',
            'desc': 'description',
            'summary': 'description',
            'overview': 'description',
            
            # Category variations
            'main_category': 'category',
            'category_slug': 'category',
            'parent_category': 'category',
            
            # URL variations
            'seller_website': 'url',
            'website': 'url',
            
            # ID variations
            'software_product_id': 'product_id',
            'technology_id': 'product_id',
            'id': 'product_id'
        }
        
        df_copy = df.copy()
        
        # Clean column names and handle duplicates
        df_copy.columns = df_copy.columns.str.lower().str.strip()
        
        # Handle duplicate columns by adding suffix
        cols = df_copy.columns.tolist()
        seen = {}
        new_cols = []
        for col in cols:
            if col in seen:
                seen[col] += 1
                new_cols.append(f"{col}_{seen[col]}")
            else:
                seen[col] = 0
                new_cols.append(col)
        df_copy.columns = new_cols
        
        # Rename columns based on mapping
        df_copy = df_copy.rename(columns=column_mapping)
        
        return df_copy
    
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and preprocess data.
        
        Args:
            df: Input dataframe
            
        Returns:
            Cleaned dataframe
        """
        df_clean = df.copy()
        
        # Clean text fields
        text_columns = ['name', 'description', 'overview']
        for col in text_columns:
            if col in df_clean.columns:
                try:
                    df_clean[col] = df_clean[col].apply(self._clean_text)
                except Exception as e:
                    self.logger.warning(f"Error cleaning column {col}: {str(e)}")
                    # Fallback: convert to string and fill missing values
                    df_clean[col] = df_clean[col].astype(str).fillna('')
        
        # Clean numeric fields
        numeric_columns = ['jobs', 'companies', 'companies_found_last_week']
        for col in numeric_columns:
            if col in df_clean.columns:
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
        
        # Remove completely empty rows
        df_clean = df_clean.dropna(how='all')
        
        # Add processed timestamp
        df_clean['processed_at'] = pd.Timestamp.now()
        
        return df_clean
    
    def _clean_text(self, text) -> str:
        """
        Clean individual text field.
        
        Args:
            text: Input text (can be string, float, int, or None)
            
        Returns:
            Cleaned text
        """
        # Handle various input types
        try:
            if text is None:
                return ''
            
            # Check if it's a pandas Series (shouldn't happen but let's be safe)
            if hasattr(text, 'iloc'):
                text = text.iloc[0] if len(text) > 0 else ''
            
            # Check for NaN values
            if pd.isna(text):
                return ''
            
            # Convert to string
            text_str = str(text)
            
            # Check for pandas 'nan' string representation
            if text_str.lower() in ['nan', 'none', 'null']:
                return ''
            
            # Remove special characters and normalize whitespace
            text_str = re.sub(r'[^\w\s\.\-\(\)]', ' ', text_str)
            text_str = re.sub(r'\s+', ' ', text_str)
            text_str = text_str.strip()
            
            # Handle truncated text indicators
            if text_str.endswith('...[Truncated]'):
                text_str = text_str.replace('...[Truncated]', '').strip()
            
            return text_str
            
        except Exception as e:
            # Fallback for any unexpected errors
            return str(text) if text is not None else ''
    
    def _find_common_columns(self, dfs: List[pd.DataFrame]) -> List[str]:
        """
        Find common columns across all datasets.
        
        Args:
            dfs: List of dataframes
            
        Returns:
            List of common column names
        """
        if not dfs:
            return []
        
        common_cols = set(dfs[0].columns)
        for df in dfs[1:]:
            common_cols = common_cols.intersection(set(df.columns))
        
        # Prioritize important columns for deduplication and output
        priority_columns = ['name', 'description', 'url', 'category', 'source']
        common_cols_list = []
        
        # Add priority columns first
        for col in priority_columns:
            if col in common_cols:
                common_cols_list.append(col)
        
        # Add remaining columns
        for col in sorted(common_cols):
            if col not in common_cols_list:
                common_cols_list.append(col)
        
        return common_cols_list
    
    def _combine_datasets(self, dfs: List[pd.DataFrame], common_columns: List[str]) -> pd.DataFrame:
        """
        Combine datasets using common columns.
        
        Args:
            dfs: List of dataframes
            common_columns: List of common column names
            
        Returns:
            Combined dataframe
        """
        if not dfs or not common_columns:
            return pd.DataFrame()
        
        # Select only common columns from each dataframe and ensure no duplicate columns
        filtered_dfs = []
        for i, df in enumerate(dfs):
            available_cols = [col for col in common_columns if col in df.columns]
            filtered_df = df[available_cols].copy()
            
            # Reset index to avoid duplicate index issues
            filtered_df = filtered_df.reset_index(drop=True)
            
            # Ensure column names are unique within this dataframe
            filtered_df.columns = pd.Index(filtered_df.columns).drop_duplicates()
            
            filtered_dfs.append(filtered_df)
        
        # Combine all dataframes
        try:
            combined_df = pd.concat(filtered_dfs, ignore_index=True, sort=False)
        except Exception as e:
            self.logger.error(f"Error combining datasets: {str(e)}")
            # Fallback: combine manually
            combined_df = pd.DataFrame()
            for df in filtered_dfs:
                if combined_df.empty:
                    combined_df = df.copy()
                else:
                    # Align columns and append
                    for col in df.columns:
                        if col not in combined_df.columns:
                            combined_df[col] = pd.NA
                    for col in combined_df.columns:
                        if col not in df.columns:
                            df = df.copy()
                            df[col] = pd.NA
                    combined_df = pd.concat([combined_df, df], ignore_index=True)
        
        # Add unique identifier
        combined_df['unique_id'] = range(len(combined_df))
        
        return combined_df
    
    def _analyze_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Analyze missing values in the dataset.
        
        Args:
            df: Input dataframe
            
        Returns:
            Dataframe with missing value statistics
        """
        missing_stats = []
        
        for col in df.columns:
            missing_count = df[col].isna().sum()
            missing_pct = (missing_count / len(df)) * 100
            
            if missing_count > 0:
                missing_stats.append({
                    'column': col,
                    'missing_count': missing_count,
                    'missing_percentage': missing_pct,
                    'total_rows': len(df)
                })
        
        return pd.DataFrame(missing_stats)
    
    def _generate_column_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate descriptive statistics for columns.
        
        Args:
            df: Input dataframe
            
        Returns:
            Dataframe with column statistics
        """
        stats = []
        
        for col in df.columns:
            col_stats = {
                'column': col,
                'data_type': str(df[col].dtype),
                'non_null_count': df[col].count(),
                'null_count': df[col].isna().sum(),
                'unique_values': df[col].nunique()
            }
            
            # Add specific stats based on data type
            if df[col].dtype in ['object', 'string']:
                col_stats['avg_length'] = df[col].astype(str).str.len().mean()
                col_stats['max_length'] = df[col].astype(str).str.len().max()
            elif pd.api.types.is_numeric_dtype(df[col]):
                col_stats['mean'] = df[col].mean()
                col_stats['std'] = df[col].std()
                col_stats['min'] = df[col].min()
                col_stats['max'] = df[col].max()
            
            stats.append(col_stats)
        
        return pd.DataFrame(stats)
