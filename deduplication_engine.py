import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
import logging
from rapidfuzz import fuzz, process
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
from collections import defaultdict

class DeduplicationEngine:
    """
    Advanced deduplication engine using multiple similarity algorithms.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.vectorizer = None
        
    def find_duplicates(self, df: pd.DataFrame, config: Dict[str, Any], progress_callback=None) -> List[Dict[str, Any]]:
        """
        Find duplicate products using specified configuration.
        
        Args:
            df: Input dataframe with product data
            config: Configuration dictionary with similarity settings
            progress_callback: Optional callback function for progress updates
            
        Returns:
            List of duplicate pairs with similarity scores
        """
        duplicates = []
        
        try:
            # Extract configuration
            threshold = config.get('similarity_threshold', 0.8)
            algorithm = config.get('algorithm', 'fuzzy_ratio')
            primary_field = config.get('primary_field', 'name')
            use_secondary = config.get('use_secondary_fields', True)
            
            self.logger.info(f"Finding duplicates with {algorithm} algorithm, threshold: {threshold}")
            
            # Prepare data for comparison
            df_clean = self._prepare_comparison_data(df, primary_field)
            
            # Get similarity function
            similarity_func = self._get_similarity_function(algorithm)
            
            # Compare all pairs
            total_comparisons = len(df_clean) * (len(df_clean) - 1) // 2
            self.logger.info(f"Performing {total_comparisons} comparisons")
            
            comparison_count = 0
            
            for i in range(len(df_clean)):
                for j in range(i + 1, len(df_clean)):
                    comparison_count += 1
                    
                    # Update progress if callback provided
                    if progress_callback and comparison_count % 100 == 0:
                        progress = (comparison_count / total_comparisons) * 100
                        progress_callback(progress, f"Comparing products... {comparison_count}/{total_comparisons}")
                    
                    # Skip exact same product (same index)
                    if i == j:
                        continue
                    
                    # Calculate primary similarity
                    similarity = similarity_func(
                        df_clean.iloc[i][primary_field],
                        df_clean.iloc[j][primary_field]
                    )
                    
                    # Apply secondary field confirmation if enabled
                    if use_secondary and similarity >= threshold * 0.9:
                        secondary_sim = self._calculate_secondary_similarity(
                            df_clean.iloc[i], df_clean.iloc[j], algorithm
                        )
                        similarity = (similarity + secondary_sim) / 2
                    
                    # Add to duplicates if above threshold
                    if similarity >= threshold:
                        duplicate_pair = {
                            'product1': df_clean.iloc[i].to_dict(),
                            'product2': df_clean.iloc[j].to_dict(),
                            'similarity': similarity,
                            'algorithm': algorithm,
                            'primary_field': primary_field,
                            'index1': i,
                            'index2': j
                        }
                        duplicates.append(duplicate_pair)
            
            # Final progress update
            if progress_callback:
                progress_callback(100, f"Completed! Found {len(duplicates)} duplicate pairs")
            
            # Sort by similarity score (highest first)
            duplicates.sort(key=lambda x: x['similarity'], reverse=True)
            
            self.logger.info(f"Found {len(duplicates)} duplicate pairs")
            
        except Exception as e:
            self.logger.error(f"Error finding duplicates: {str(e)}")
            raise
        
        return duplicates
    
    def create_master_catalogue(self, df: pd.DataFrame, duplicates: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Create master catalogue by merging duplicate products.
        
        Args:
            df: Original dataframe
            duplicates: List of duplicate pairs
            
        Returns:
            Master catalogue dataframe
        """
        try:
            # Create a copy of the original dataframe
            master_df = df.copy()
            
            # Track which rows to remove (duplicates)
            rows_to_remove = set()
            
            # Group duplicates by connected components
            duplicate_groups = self._group_duplicates(duplicates)
            
            self.logger.info(f"Found {len(duplicate_groups)} duplicate groups")
            
            # Merge each group of duplicates
            for group in duplicate_groups:
                if len(group) < 2:
                    continue
                
                # Find the best representative (highest data completeness)
                representative_idx = self._select_representative(master_df, group)
                
                # Merge information from all duplicates into representative
                merged_record = self._merge_duplicate_records(master_df, group, representative_idx)
                
                # Update the representative record
                for col, value in merged_record.items():
                    if col in master_df.columns:
                        master_df.at[representative_idx, col] = value
                
                # Mark other records for removal
                for idx in group:
                    if idx != representative_idx:
                        rows_to_remove.add(idx)
            
            # Remove duplicate rows
            master_df = master_df.drop(index=list(rows_to_remove))
            master_df = master_df.reset_index(drop=True)
            
            # Filter to only include required output columns
            required_columns = ['name', 'description', 'url', 'category', 'source']
            output_columns = []
            
            for col in required_columns:
                if col in master_df.columns:
                    output_columns.append(col)
            
            # Keep only the required columns in the final output
            if output_columns:
                master_df = master_df[output_columns].copy()
            
            # Rename 'name' to 'product_name' for clarity in output
            if 'name' in master_df.columns:
                master_df = master_df.rename(columns={'name': 'product_name'})
            
            # Add metadata
            master_df['catalogue_created_at'] = pd.Timestamp.now()
            
            self.logger.info(f"Created master catalogue with {len(master_df)} unique products")
            
        except Exception as e:
            self.logger.error(f"Error creating master catalogue: {str(e)}")
            raise
        
        return master_df
    
    def _prepare_comparison_data(self, df: pd.DataFrame, primary_field: str) -> pd.DataFrame:
        """
        Prepare data for comparison by cleaning and normalizing text fields.
        
        Args:
            df: Input dataframe
            primary_field: Primary field for comparison
            
        Returns:
            Prepared dataframe
        """
        df_clean = df.copy()
        
        # Create combined field if requested
        if primary_field == 'combined':
            df_clean['combined'] = df_clean.apply(
                lambda row: f"{row.get('name', '')} {row.get('description', '')}", axis=1
            )
        
        # Clean and normalize text fields
        text_fields = ['name', 'description', 'combined']
        for field in text_fields:
            if field in df_clean.columns:
                df_clean[field] = df_clean[field].astype(str).apply(self._normalize_text)
        
        return df_clean
    
    def _normalize_text(self, text: str) -> str:
        """
        Normalize text for comparison.
        
        Args:
            text: Input text
            
        Returns:
            Normalized text
        """
        if pd.isna(text) or text == 'nan':
            return ''
        
        # Convert to lowercase
        text = str(text).lower()
        
        # Remove special characters and extra whitespace
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        # Remove common stop words that don't add value for matching
        stop_words = ['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by']
        words = text.split()
        words = [word for word in words if word not in stop_words]
        
        return ' '.join(words)
    
    def _get_similarity_function(self, algorithm: str):
        """
        Get similarity function based on algorithm name.
        
        Args:
            algorithm: Algorithm name
            
        Returns:
            Similarity function
        """
        if algorithm == 'fuzzy_ratio':
            return lambda x, y: fuzz.ratio(x, y) / 100.0
        elif algorithm == 'fuzzy_partial':
            return lambda x, y: fuzz.partial_ratio(x, y) / 100.0
        elif algorithm == 'fuzzy_token_sort':
            return lambda x, y: fuzz.token_sort_ratio(x, y) / 100.0
        elif algorithm == 'tfidf_cosine':
            return self._tfidf_similarity
        else:
            return lambda x, y: fuzz.ratio(x, y) / 100.0
    
    def _tfidf_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate TF-IDF cosine similarity between two texts.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score between 0 and 1
        """
        try:
            if not text1 or not text2:
                return 0.0
            
            # Initialize vectorizer if not already done
            if self.vectorizer is None:
                self.vectorizer = TfidfVectorizer(
                    ngram_range=(1, 2),
                    min_df=1,
                    stop_words='english'
                )
            
            # Vectorize texts
            vectors = self.vectorizer.fit_transform([text1, text2])
            
            # Calculate cosine similarity
            similarity_matrix = cosine_similarity(vectors)
            return similarity_matrix[0, 1]
            
        except Exception as e:
            self.logger.warning(f"Error calculating TF-IDF similarity: {str(e)}")
            return 0.0
    
    def _calculate_secondary_similarity(self, record1: pd.Series, record2: pd.Series, algorithm: str) -> float:
        """
        Calculate similarity using secondary fields for confirmation.
        
        Args:
            record1: First product record
            record2: Second product record
            algorithm: Algorithm to use
            
        Returns:
            Secondary similarity score
        """
        similarity_func = self._get_similarity_function(algorithm)
        
        # Fields to check for secondary confirmation
        secondary_fields = ['description', 'category', 'url']
        similarities = []
        
        for field in secondary_fields:
            if field in record1.index and field in record2.index:
                try:
                    val1_raw = record1[field] 
                    val2_raw = record2[field]
                    val1 = str(val1_raw) if val1_raw is not None and not (isinstance(val1_raw, float) and np.isnan(val1_raw)) else ''
                    val2 = str(val2_raw) if val2_raw is not None and not (isinstance(val2_raw, float) and np.isnan(val2_raw)) else ''
                    
                    if val1.strip() != '' and val2.strip() != '':
                        sim = similarity_func(val1, val2)
                        similarities.append(float(sim))
                except Exception as e:
                    # Skip problematic field comparisons
                    continue
        
        # Return average similarity or 0 if no valid comparisons
        return float(np.mean(similarities)) if similarities else 0.0
    
    def _group_duplicates(self, duplicates: List[Dict[str, Any]]) -> List[List[int]]:
        """
        Group duplicates by connected components (transitive closure).
        
        Args:
            duplicates: List of duplicate pairs
            
        Returns:
            List of duplicate groups (each group is a list of indices)
        """
        # Build adjacency list
        graph = defaultdict(set)
        all_indices = set()
        
        for dup in duplicates:
            idx1 = dup['index1']
            idx2 = dup['index2']
            graph[idx1].add(idx2)
            graph[idx2].add(idx1)
            all_indices.add(idx1)
            all_indices.add(idx2)
        
        # Find connected components using DFS
        visited = set()
        groups = []
        
        for idx in all_indices:
            if idx not in visited:
                group = []
                stack = [idx]
                
                while stack:
                    current = stack.pop()
                    if current not in visited:
                        visited.add(current)
                        group.append(current)
                        stack.extend(graph[current] - visited)
                
                if group:
                    groups.append(group)
        
        return groups
    
    def _select_representative(self, df: pd.DataFrame, group: List[int]) -> int:
        """
        Select the best representative from a group of duplicates.
        
        Args:
            df: Dataframe
            group: List of indices in the duplicate group
            
        Returns:
            Index of the selected representative
        """
        best_idx = group[0]
        best_score = 0
        
        for idx in group:
            # Calculate completeness score
            record = df.iloc[idx]
            completeness = record.notna().sum() / len(record)
            
            # Bonus for longer descriptions
            desc_length = len(str(record.get('description', '')))
            length_bonus = min(desc_length / 1000, 0.2)  # Max 20% bonus
            
            # Combine scores
            total_score = completeness + length_bonus
            
            if total_score > best_score:
                best_score = total_score
                best_idx = idx
        
        return best_idx
    
    def _merge_duplicate_records(self, df: pd.DataFrame, group: List[int], representative_idx: int) -> Dict[str, Any]:
        """
        Merge information from duplicate records into a single record.
        
        Args:
            df: Dataframe
            group: List of indices to merge
            representative_idx: Index of the representative record
            
        Returns:
            Merged record as dictionary
        """
        merged_record = df.iloc[representative_idx].to_dict()
        
        # Merge strategy for different field types
        for col in df.columns:
            values = []
            
            for idx in group:
                value = df.iloc[idx][col]
                if pd.notna(value) and str(value).strip():
                    values.append(str(value))
            
            if not values:
                continue
            
            # Field-specific merge strategies
            if col in ['name']:
                # Use the shortest non-empty name (usually cleaner)
                merged_record[col] = min(values, key=len) if values else merged_record[col]
            
            elif col in ['description', 'overview']:
                # Use the longest description (more information)
                merged_record[col] = max(values, key=len) if values else merged_record[col]
            
            elif col in ['url']:
                # Prefer non-empty URLs
                non_empty_urls = [v for v in values if v and v != 'nan']
                if non_empty_urls:
                    merged_record[col] = non_empty_urls[0]
            
            elif col in ['category']:
                # Use most common category or first non-empty
                if values:
                    merged_record[col] = max(set(values), key=values.count)
            
            # For other fields, keep the representative's value or use first non-empty
            elif pd.isna(merged_record[col]) or not str(merged_record[col]).strip():
                if values:
                    merged_record[col] = values[0]
        
        # Add merge metadata
        merged_record['merged_from_sources'] = ','.join(set(
            str(df.iloc[idx].get('source', '')) for idx in group
        ))
        merged_record['merge_count'] = len(group)
        
        return merged_record
