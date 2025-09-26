from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
import pandas as pd
import numpy as np
import io
import os
from typing import Dict, Any, List
import json
from datetime import datetime
import uvicorn

app = FastAPI(title="CSV Indexer API", description="Generate comprehensive index files for CSV data analysis")

def detect_column_types(df: pd.DataFrame) -> Dict[str, str]:
    """Detect and categorize column types"""
    column_types = {}
    
    for col in df.columns:
        # Skip empty columns
        if df[col].isna().all():
            column_types[col] = "empty"
            continue
            
        # Try to convert to numeric
        numeric_series = pd.to_numeric(df[col], errors='coerce')
        
        if not numeric_series.isna().all():
            # Check if it's integer or float
            if numeric_series.dropna().apply(lambda x: x.is_integer()).all():
                column_types[col] = "integer"
            else:
                column_types[col] = "float"
        else:
            # Check if it's datetime
            try:
                pd.to_datetime(df[col], errors='raise')
                column_types[col] = "datetime"
            except:
                # Check unique values to determine if categorical
                unique_count = df[col].nunique()
                total_count = len(df[col].dropna())
                
                if unique_count <= 20 or (unique_count / total_count) < 0.1:
                    column_types[col] = "categorical"
                else:
                    column_types[col] = "text"
    
    return column_types

def get_sample_rows(df: pd.DataFrame, n_samples: int = 3) -> List[Dict]:
    """Get representative sample rows"""
    # Filter out completely empty rows
    non_empty_df = df.dropna(how='all')

    if len(non_empty_df) <= n_samples:
        return non_empty_df.to_dict('records')

    # Get a mix of random samples and edge cases
    sample_indices = []

    # Add first and last non-empty rows
    sample_indices.extend([0, len(non_empty_df) - 1])

    # Add random samples
    remaining_samples = n_samples - 2
    if remaining_samples > 0 and len(non_empty_df) > 2:
        random_indices = np.random.choice(
            range(1, len(non_empty_df) - 1),
            size=min(remaining_samples, len(non_empty_df) - 2),
            replace=False
        )
        sample_indices.extend(random_indices)

    return non_empty_df.iloc[sample_indices].to_dict('records')

def get_numeric_stats(df: pd.DataFrame, column_types: Dict[str, str]) -> Dict[str, Dict]:
    """Calculate comprehensive statistics for numeric columns"""
    numeric_stats = {}
    
    for col, col_type in column_types.items():
        if col_type in ['integer', 'float']:
            series = pd.to_numeric(df[col], errors='coerce').dropna()
            
            if len(series) > 0:
                stats = {
                    'count': len(series),
                    'mean': round(series.mean(), 4),
                    'median': round(series.median(), 4),
                    'std': round(series.std(), 4),
                    'min': round(series.min(), 4),
                    'max': round(series.max(), 4),
                    'q25': round(series.quantile(0.25), 4),
                    'q75': round(series.quantile(0.75), 4),
                    'null_count': df[col].isna().sum(),
                    'null_percentage': round((df[col].isna().sum() / len(df)) * 100, 2)
                }
                
                # Add distribution insights
                if stats['std'] > 0:
                    stats['coefficient_of_variation'] = round(stats['std'] / stats['mean'], 4)
                
                numeric_stats[col] = stats
    
    return numeric_stats

def get_categorical_info(df: pd.DataFrame, column_types: Dict[str, str]) -> Dict[str, Dict]:
    """Get information about categorical columns"""
    categorical_info = {}
    
    for col, col_type in column_types.items():
        if col_type == 'categorical':
            value_counts = df[col].value_counts()
            
            categorical_info[col] = {
                'unique_count': df[col].nunique(),
                'null_count': df[col].isna().sum(),
                'null_percentage': round((df[col].isna().sum() / len(df)) * 100, 2),
                'top_values': value_counts.head(10).to_dict(),
                'most_common': value_counts.index[0] if len(value_counts) > 0 else None,
                'most_common_count': value_counts.iloc[0] if len(value_counts) > 0 else 0
            }
    
    return categorical_info

def analyze_data_quality(df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze overall data quality"""
    total_cells = df.shape[0] * df.shape[1]
    null_cells = df.isna().sum().sum()
    
    return {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'total_cells': total_cells,
        'null_cells': null_cells,
        'null_percentage': round((null_cells / total_cells) * 100, 2),
        'duplicate_rows': df.duplicated().sum(),
        'memory_usage_mb': round(df.memory_usage(deep=True).sum() / 1024 / 1024, 2)
    }

def generate_insights(df: pd.DataFrame, column_types: Dict[str, str], 
                     numeric_stats: Dict, categorical_info: Dict) -> List[str]:
    """Generate automated insights about the dataset"""
    insights = []
    
    # Dataset size insights
    insights.append(f"Dataset contains {len(df):,} rows and {len(df.columns)} columns")
    
    # Column type distribution
    type_counts = {}
    for col_type in column_types.values():
        type_counts[col_type] = type_counts.get(col_type, 0) + 1
    
    insights.append(f"Column types: {dict(type_counts)}")
    
    # Missing data insights
    null_percentages = [(col, (df[col].isna().sum() / len(df)) * 100) 
                       for col in df.columns if df[col].isna().sum() > 0]
    
    if null_percentages:
        high_null_cols = [col for col, pct in null_percentages if pct > 50]
        if high_null_cols:
            insights.append(f"Columns with >50% missing data: {high_null_cols}")
    
    # Numeric insights
    if numeric_stats:
        high_var_cols = [col for col, stats in numeric_stats.items() 
                        if stats.get('coefficient_of_variation', 0) > 1]
        if high_var_cols:
            insights.append(f"Highly variable numeric columns: {high_var_cols}")
    
    return insights

@app.post("/index-csv/")
async def index_csv(file: UploadFile = File(...)):
    """
    Upload a CSV file and generate a comprehensive index file
    """
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="File must be a CSV")
    
    try:
        # Read CSV file
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        
        # Generate analysis
        column_types = detect_column_types(df)
        sample_rows = get_sample_rows(df)
        numeric_stats = get_numeric_stats(df, column_types)
        categorical_info = get_categorical_info(df, column_types)
        data_quality = analyze_data_quality(df)
        insights = generate_insights(df, column_types, numeric_stats, categorical_info)
        
        # Generate index content
        index_content = generate_index_content(
            file.filename, df, column_types, sample_rows, 
            numeric_stats, categorical_info, data_quality, insights
        )
        
        # Save index file
        index_filename = f"index_{file.filename.replace('.csv', '.txt')}"
        index_path = index_filename  # Save in current directory

        with open(index_path, 'w', encoding='utf-8') as f:
            f.write(index_content)
        
        return {
            "message": "CSV indexed successfully",
            "index_file": index_filename,
            "summary": {
                "rows": len(df),
                "columns": len(df.columns),
                "file_size_mb": round(len(contents) / 1024 / 1024, 2)
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing CSV: {str(e)}")

def generate_index_content(filename: str, df: pd.DataFrame, column_types: Dict[str, str],
                          sample_rows: List[Dict], numeric_stats: Dict, 
                          categorical_info: Dict, data_quality: Dict, insights: List[str]) -> str:
    """Generate the comprehensive index file content"""
    
    content = f"""# CSV INDEX FILE
# Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
# Original file: {filename}

## DATASET OVERVIEW
{data_quality['total_rows']:,} rows × {data_quality['total_columns']} columns
File size: {data_quality['memory_usage_mb']} MB
Missing data: {data_quality['null_percentage']}% of total cells
Duplicate rows: {data_quality['duplicate_rows']:,}

## KEY INSIGHTS
"""
    
    for insight in insights:
        content += f"- {insight}\n"
    
    content += f"""
## COLUMN SCHEMA
"""
    
    for i, (col, col_type) in enumerate(column_types.items(), 1):
        null_count = df[col].isna().sum()
        null_pct = round((null_count / len(df)) * 100, 2)
        content += f"{i:2d}. {col}\n"
        content += f"    Type: {col_type}\n"
        content += f"    Missing: {null_count:,} ({null_pct}%)\n"
        
        if col_type in ['integer', 'float'] and col in numeric_stats:
            stats = numeric_stats[col]
            content += f"    Range: {stats['min']} to {stats['max']}\n"
            content += f"    Mean: {stats['mean']}, Median: {stats['median']}\n"
        elif col_type == 'categorical' and col in categorical_info:
            info = categorical_info[col]
            content += f"    Unique values: {info['unique_count']}\n"
            content += f"    Most common: '{info['most_common']}' ({info['most_common_count']} times)\n"
        
        content += "\n"
    
    content += "## SAMPLE DATA\n"
    for i, row in enumerate(sample_rows, 1):
        content += f"Row {i}:\n"
        for col, value in row.items():
            content += f"  {col}: {value}\n"
        content += "\n"
    
    if numeric_stats:
        content += "## NUMERIC STATISTICS\n"
        for col, stats in numeric_stats.items():
            content += f"{col}:\n"
            content += f"  Count: {stats['count']:,}\n"
            content += f"  Mean: {stats['mean']}\n"
            content += f"  Median: {stats['median']}\n"
            content += f"  Std Dev: {stats['std']}\n"
            content += f"  Min: {stats['min']}\n"
            content += f"  Max: {stats['max']}\n"
            content += f"  Q1: {stats['q25']}\n"
            content += f"  Q3: {stats['q75']}\n"
            if 'coefficient_of_variation' in stats:
                content += f"  Coefficient of Variation: {stats['coefficient_of_variation']}\n"
            content += "\n"
    
    if categorical_info:
        content += "## CATEGORICAL DATA\n"
        for col, info in categorical_info.items():
            content += f"{col}:\n"
            content += f"  Unique values: {info['unique_count']}\n"
            content += f"  Top values:\n"
            for value, count in info['top_values'].items():
                content += f"    '{value}': {count}\n"
            content += "\n"
    
    content += """## USAGE NOTES
- This index provides comprehensive metadata about your CSV file
- Use column names exactly as shown in the schema section
- Pay attention to data types when writing analysis code
- Consider missing data percentages when performing calculations
- Sample data shows actual values from your dataset

## FOR LLM CODE GENERATION
When generating Python code for this dataset:
1. Use pandas to load the CSV: df = pd.read_csv('filename.csv')
2. Handle missing values appropriately based on null percentages shown
3. Use correct data types as specified in the schema
4. Reference the sample data to understand value formats
5. Consider the data distribution when choosing analysis methods
"""
    
    return content

@app.get("/download-index/{filename}")
async def download_index(filename: str):
    """Download the generated index file"""
    file_path = f"index_{filename}"
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Index file not found")

    return FileResponse(
        path=file_path,
        filename=f"index_{filename}",
        media_type='text/plain'
    )

@app.get("/")
async def root():
    return {"message": "CSV Indexer API is running"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
