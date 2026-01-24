# Data Analysis Scripts

This directory contains utility scripts for data validation, analysis, and preprocessing.

## Available Scripts

### 1. `validate_data.py`
Validates the tunnel squeezing dataset for completeness and quality.

**Usage:**
```bash
# Validate default data file
python scripts/validate_data.py

# Validate custom data file
python scripts/validate_data.py path/to/data.csv
```

**Checks performed:**
- File existence and readability
- Required columns presence
- Data types validation
- Missing values detection
- Valid value ranges (positive values for D, H, Q, K)
- Valid class labels (1, 2, or 3)
- Duplicate rows detection
- Outlier warnings

**Output:**
- ✅ Success message if validation passes
- ❌ Error messages with specific issues if validation fails
- Returns exit code 1 on failure (useful for CI/CD pipelines)

### 2. `analyze_data.py`
Generates comprehensive data analysis and quality report.

**Usage:**
```bash
# Analyze default data file
python scripts/analyze_data.py

# Analyze custom data file
python scripts/analyze_data.py path/to/data.csv
```

**Report sections:**
- Basic information (sample count, features)
- Data types
- Missing values analysis
- Class distribution with visualization
- Feature statistics (mean, std, min, max, quartiles)
- Feature ranges and outlier detection
- Feature correlations with target class
- Duplicate detection
- Overall data quality score

**Output:**
- Detailed console report with statistics and visualizations
- Quality score (0-100) with recommendations

## Example Workflow

```bash
# Step 1: Validate the data
python scripts/validate_data.py

# Step 2: Generate analysis report
python scripts/analyze_data.py

# Step 3: If validation passes, proceed with model training
python -m jupyter notebook Tunnel_Squeezing_RandomForest.ipynb
```

## Integration with CI/CD

These scripts can be integrated into automated pipelines:

```yaml
# Example CI configuration
- name: Validate Data
  run: python scripts/validate_data.py
  
- name: Generate Data Report
  run: python scripts/analyze_data.py > data_report.txt
```

## Dependencies

- pandas
- numpy

Install dependencies:
```bash
pip install -r requirements.txt
```

## Adding New Scripts

When adding new analysis scripts:
1. Follow the existing naming convention
2. Include comprehensive docstrings
3. Support command-line arguments for file paths
4. Provide clear output messages
5. Update this README with usage instructions
