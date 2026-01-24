"""
Generate data statistics and quality report.
"""
import pandas as pd
import numpy as np
import sys


def generate_data_report(filepath: str):
    """Generate comprehensive data analysis report."""
    
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"❌ Error: File not found: {filepath}")
        sys.exit(1)
    except pd.errors.EmptyDataError:
        print(f"❌ Error: CSV file is empty")
        sys.exit(1)
    except pd.errors.ParserError as e:
        print(f"❌ Error parsing CSV file: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error loading file: {e}")
        sys.exit(1)
    
    print("=" * 70)
    print("TUNNEL SQUEEZING DATASET REPORT")
    print("=" * 70)
    print()
    
    # Basic Information
    print("📊 BASIC INFORMATION")
    print("-" * 70)
    print(f"Total samples: {len(df)}")
    print(f"Total features: {len(df.columns)}")
    print(f"Features: {list(df.columns)}")
    print()
    
    # Data Types
    print("📋 DATA TYPES")
    print("-" * 70)
    for col, dtype in df.dtypes.items():
        print(f"  {col:<20} {dtype}")
    print()
    
    # Missing Values
    print("🔍 MISSING VALUES")
    print("-" * 70)
    missing = df.isnull().sum()
    if missing.sum() == 0:
        print("  ✅ No missing values found!")
    else:
        for col, count in missing[missing > 0].items():
            percentage = (count / len(df)) * 100
            print(f"  {col}: {count} ({percentage:.2f}%)")
    print()
    
    # Class Distribution
    print("📈 CLASS DISTRIBUTION")
    print("-" * 70)
    if 'Class' in df.columns:
        class_counts = df['Class'].value_counts().sort_index()
        class_names = {1: "Non-squeezing", 2: "Minor squeezing", 3: "Severe squeezing"}
        
        for cls, count in class_counts.items():
            percentage = (count / len(df)) * 100
            class_name = class_names.get(cls, f"Class {cls}")
            bar = "█" * int(percentage / 2)
            print(f"  Class {cls} ({class_name:<18}): {count:3d} ({percentage:5.1f}%) {bar}")
        
        # Check for imbalance
        max_count = class_counts.max()
        min_count = class_counts.min()
        imbalance_ratio = max_count / min_count
        print()
        print(f"  Imbalance ratio: {imbalance_ratio:.2f}:1")
        if imbalance_ratio > 2:
            print("  ⚠️  Dataset is imbalanced - consider using SMOTE or class weights")
    print()
    
    # Feature Statistics
    print("📊 FEATURE STATISTICS")
    print("-" * 70)
    
    # Select numeric columns (excluding 'No' if it exists)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'No' in numeric_cols:
        numeric_cols.remove('No')
    
    if numeric_cols:
        stats = df[numeric_cols].describe()
        print(stats.to_string())
    print()
    
    # Feature Ranges and Outliers
    print("📏 FEATURE RANGES & OUTLIERS")
    print("-" * 70)
    feature_cols = ['D (m)', 'H(m)', 'Q', 'K(MPa)']
    
    for col in feature_cols:
        if col in df.columns:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            outliers_low = (df[col] < lower_bound).sum()
            outliers_high = (df[col] > upper_bound).sum()
            total_outliers = outliers_low + outliers_high
            
            print(f"  {col:<15} Range: [{df[col].min():.3f}, {df[col].max():.3f}]")
            if total_outliers > 0:
                print(f"                  Outliers: {total_outliers} ({outliers_low} low, {outliers_high} high)")
    print()
    
    # Correlations
    print("🔗 FEATURE CORRELATIONS (with Class)")
    print("-" * 70)
    if 'Class' in df.columns and len(feature_cols) > 0:
        existing_features = [col for col in feature_cols if col in df.columns]
        correlations = df[existing_features + ['Class']].corr()['Class'].sort_values(ascending=False)
        
        for feature, corr in correlations.items():
            if feature != 'Class':
                corr_bar = "█" * int(abs(corr) * 20)
                sign = "+" if corr > 0 else "-"
                print(f"  {feature:<15} {sign}{abs(corr):.3f} {corr_bar}")
    print()
    
    # Duplicates
    print("🔄 DUPLICATE DETECTION")
    print("-" * 70)
    duplicates = df.duplicated().sum()
    if duplicates == 0:
        print("  ✅ No duplicate rows found!")
    else:
        print(f"  ⚠️  Found {duplicates} duplicate rows")
        print(f"     Consider removing duplicates to improve data quality")
    print()
    
    # Data Quality Summary
    print("✨ DATA QUALITY SUMMARY")
    print("-" * 70)
    quality_score = 100
    issues = []
    
    if missing.sum() > 0:
        quality_score -= 20
        issues.append("Missing values detected")
    
    if duplicates > 0:
        quality_score -= 10
        issues.append("Duplicate rows detected")
    
    if 'Class' in df.columns:
        class_counts = df['Class'].value_counts()
        if class_counts.max() / class_counts.min() > 2:
            quality_score -= 15
            issues.append("Class imbalance detected")
    
    # Check for outliers
    total_outliers_all = 0
    for col in feature_cols:
        if col in df.columns:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            outliers = ((df[col] < q1 - 1.5 * iqr) | (df[col] > q3 + 1.5 * iqr)).sum()
            total_outliers_all += outliers
    
    if total_outliers_all > len(df) * 0.05:  # More than 5% outliers
        quality_score -= 10
        issues.append("Significant outliers detected")
    
    print(f"  Overall Quality Score: {quality_score}/100")
    
    if quality_score >= 90:
        print(f"  ✅ Excellent data quality!")
    elif quality_score >= 70:
        print(f"  ⚠️  Good data quality with minor issues")
    else:
        print(f"  ❌ Data quality needs improvement")
    
    if issues:
        print(f"\n  Issues found:")
        for issue in issues:
            print(f"    - {issue}")
    
    print()
    print("=" * 70)


if __name__ == "__main__":
    # Default path
    filepath = 'data/raw/tunnel.csv'
    
    # Allow custom path from command line
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
    
    generate_data_report(filepath)
