# Product Catalogue Deduplication System

A comprehensive Streamlit-based application for merging and deduplicating product catalogs using advanced text similarity algorithms. This system helps create master product catalogues by identifying and consolidating duplicate products across multiple data sources.

## Features

### 🔍 **Advanced Deduplication**
- Multiple similarity algorithms (Fuzzy matching, TF-IDF, Token-based)
- Configurable similarity thresholds
- Cross-dataset duplicate detection
- Intelligent conflict resolution

### 📊 **Interactive Data Exploration**
- Automated data quality analysis
- Missing value identification
- Column statistics and distribution
- Visual data profiling

### 🎯 **Smart Matching Engine**
- Fuzzy string matching using RapidFuzz
- TF-IDF cosine similarity for text comparison
- Multi-field validation for higher accuracy
- Transitive closure for duplicate grouping

### 📋 **Master Catalogue Generation**
- Intelligent record merging
- Data completeness-based representative selection
- Source tracking and metadata preservation
- Quality score calculation

### 📤 **Export & Documentation**
- CSV export of deduplicated catalogue
- Comprehensive process documentation
- Duplicate pairs analysis
- Quality metrics reporting

## Installation

### Prerequisites
- Python 3.8 or higher
- Required packages (install via pip):

```bash
pip install streamlit pandas numpy plotly rapidfuzz scikit-learn
