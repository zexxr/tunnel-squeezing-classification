"""
Data validation script for tunnel squeezing dataset.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import os


def validate_tunnel_data(filepath: str) -> Tuple[bool, List[str]]:
    """
    Validate the tunnel dataset for completeness and quality.
    
    Parameters:
        filepath: Path to the CSV file
        
    Returns:
        Tuple of (is_valid, error_messages)
    """
    errors = []
    
    # Check file exists
    if not os.path.exists(filepath):
        errors.append(f"File not found: {filepath}")
        return False, errors
    
    try:
        # Load the data
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        errors.append(f"CSV file not found: {filepath}")
        return False, errors
    except pd.errors.EmptyDataError:
        errors.append("CSV file is empty")
        return False, errors
    except pd.errors.ParserError as e:
        errors.append(f"Error parsing CSV file: {str(e)}")
        return False, errors
    except Exception as e:
        errors.append(f"Unexpected error loading CSV file: {str(e)}")
        return False, errors
    
    # Check required columns exist
    required_columns = ['D (m)', 'H(m)', 'Q', 'K(MPa)', 'Class']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        errors.append(f"Missing required columns: {missing_columns}")
    
    # Check for missing values (only in columns that exist)
    existing_required_columns = [col for col in required_columns if col in df.columns]
    if existing_required_columns:
        missing_values = df[existing_required_columns].isnull().sum()
        if missing_values.any():
            for col, count in missing_values[missing_values > 0].items():
                errors.append(f"Column '{col}' has {count} missing values")
    
    # Check data types
    expected_types = {
        'D (m)': ['float64', 'int64'],
        'H(m)': ['float64', 'int64'],
        'Q': ['float64', 'int64'],
        'K(MPa)': ['float64', 'int64'],
        'Class': ['int64']
    }
    
    for col, expected in expected_types.items():
        if col in df.columns:
            if str(df[col].dtype) not in expected:
                errors.append(f"Column '{col}' has type {df[col].dtype}, expected one of {expected}")
    
    # Check for valid ranges
    if 'D (m)' in df.columns:
        if (df['D (m)'] <= 0).any():
            errors.append("Column 'D (m)' contains non-positive values")
    
    if 'H(m)' in df.columns:
        if (df['H(m)'] <= 0).any():
            errors.append("Column 'H(m)' contains non-positive values")
    
    if 'Q' in df.columns:
        if (df['Q'] <= 0).any():
            errors.append("Column 'Q' contains non-positive values (Q must be > 0)")
    
    if 'K(MPa)' in df.columns:
        if (df['K(MPa)'] < 0).any():
            errors.append("Column 'K(MPa)' contains negative values")
    
    # Check class labels are valid (1, 2, or 3)
    if 'Class' in df.columns:
        valid_classes = {1, 2, 3}
        unique_classes = set(df['Class'].unique())
        invalid_classes = unique_classes - valid_classes
        if invalid_classes:
            errors.append(f"Invalid class labels found: {invalid_classes}. Valid classes are {valid_classes}")
    
    # Check for duplicates
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        errors.append(f"Found {duplicates} duplicate rows")
    
    # Check for reasonable value ranges (warnings, not errors)
    if len(errors) == 0:  # Only check warnings if no errors
        warnings = []
        
        if 'D (m)' in df.columns:
            if df['D (m)'].max() > 50:
                warnings.append(f"Warning: Very large tunnel diameter detected: {df['D (m)'].max():.1f} m")
        
        if 'H(m)' in df.columns:
            if df['H(m)'].max() > 5000:
                warnings.append(f"Warning: Very large overburden detected: {df['H(m)'].max():.1f} m")
        
        if 'Q' in df.columns:
            if df['Q'].max() > 10000:
                warnings.append(f"Warning: Very high Q value detected: {df['Q'].max():.3f}")
        
        # Print warnings if any
        for warning in warnings:
            print(f"  ⚠️  {warning}")
    
    return len(errors) == 0, errors


def validate_features_columns(df: pd.DataFrame) -> List[str]:
    """
    Validate that feature columns match expected format.
    
    Parameters:
        df: DataFrame to validate
        
    Returns:
        List of error messages
    """
    errors = []
    expected_features = ['D (m)', 'H(m)', 'Q', 'K(MPa)']
    
    for feature in expected_features:
        if feature not in df.columns:
            errors.append(f"Missing feature column: {feature}")
    
    return errors


if __name__ == "__main__":
    import sys
    
    # Default path
    filepath = 'data/raw/tunnel.csv'
    
    # Allow custom path from command line
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
    
    print(f"Validating data file: {filepath}\n")
    
    is_valid, errors = validate_tunnel_data(filepath)
    
    if is_valid:
        print("✅ Data validation passed!")
        
        # Only load data once for additional stats
        try:
            df = pd.read_csv(filepath)
            print("\nAdditional checks:")
            print(f"  - Total samples: {len(df)}")
            print(f"  - Features: {list(df.columns)}")
            if 'Class' in df.columns:
                print(f"  - Class distribution:")
                for cls, count in df['Class'].value_counts().sort_index().items():
                    percentage = (count / len(df)) * 100
                    print(f"    Class {cls}: {count} ({percentage:.1f}%)")
        except Exception as e:
            print(f"\nNote: Could not load additional statistics: {e}")
    else:
        print("❌ Data validation failed:")
        for error in errors:
            print(f"  - {error}")
        sys.exit(1)
