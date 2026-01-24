# Tunnel Squeezing Dataset

## Overview
Dataset for predicting tunnel squeezing severity in underground construction. This dataset contains geotechnical parameters and corresponding squeezing classifications for various tunnel construction scenarios.

## Data Dictionary

| Column | Description | Type | Unit | Range |
|--------|-------------|------|------|-------|
| No | Sample number (index) | int | - | 1-117 |
| D (m) | Tunnel diameter | float | meters | 1.0 - 20.0 |
| H(m) | Overburden depth | float | meters | 10.0 - 3000.0 |
| Q | Rock mass quality index (Barton's Q-value) | float | - | 0.001 - 1000 |
| K(MPa) | Rock mass stiffness | float | MPa | > 0 |
| ε (%) | Tunnel wall strain percentage | float | % | - |
| Class | Squeezing severity class | int | - | 1, 2, 3 |

## Class Labels
- **Class 1**: Non-squeezing (ε < 1%)
  - Stable conditions with standard support requirements
  - Minimal rock deformation expected
  
- **Class 2**: Minor squeezing (1% ≤ ε < 2.5%)
  - Moderate convergence requiring monitoring
  - Additional support may be needed
  
- **Class 3**: Severe squeezing (ε ≥ 2.5%)
  - Heavy flowing ground conditions
  - Requires yielding support systems and intensive monitoring

## Data Source
This dataset represents tunnel squeezing case studies from various underground construction projects, compiled for geotechnical risk assessment and machine learning analysis.

## Statistics
*Statistics as of January 2024. Run `python scripts/analyze_data.py` for current statistics.*

- **Total samples**: 117
- **Class distribution**:
  - Class 1 (Non-squeezing): 33 (28.2%)
  - Class 2 (Minor squeezing): 24 (20.5%)
  - Class 3 (Severe squeezing): 60 (51.3%)

## Data Quality
- **Missing values**: None (0 missing values across all columns)
- **Outliers**: Dataset has been reviewed for quality
- **Data collection period**: Historical case studies from tunnel construction projects
- **Duplicates**: No duplicate entries detected

## Usage
```python
import pandas as pd

# Load the raw dataset
df = pd.read_csv('data/raw/tunnel.csv')

# Features for model training
features = ['D (m)', 'H(m)', 'Q', 'K(MPa)']
target = 'Class'

X = df[features]
y = df[target]
```

## Notes
- The Q-value (rock mass quality) follows a logarithmic scale
- Strain percentage (ε) is included for reference but is not used as a feature for prediction
- Class distribution is imbalanced, with Class 3 (severe squeezing) being most common
- SMOTE or other balancing techniques may be beneficial for model training
