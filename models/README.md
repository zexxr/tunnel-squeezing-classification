# Trained Models

This directory contains trained machine learning models for tunnel squeezing classification.

## Available Models

### 1. Random Forest (`rf_tunnel_squeezing.pkl`)
- **Algorithm**: Random Forest Classifier
- **Training Date**: 2024
- **Performance Metrics**:
  - Accuracy: ~87%
  - Balanced Accuracy: ~85%
  - F1-macro: ~0.84
- **Key Features**:
  - Ensemble learning method using multiple decision trees
  - Robust to overfitting with proper hyperparameter tuning
  - Provides feature importance rankings
- **Input Features**: D (m), H(m), Q, K(MPa)
- **Output**: Class prediction (1, 2, or 3)
- **Hyperparameters**:
  - n_estimators: Optimized through cross-validation
  - max_depth: Controlled to prevent overfitting
  - min_samples_split: Tuned for optimal performance

### 2. SVM Enhanced (`svm_tunnel_squeezing_enhanced.pkl`)
- **Algorithm**: Support Vector Machine with RBF kernel
- **Training Date**: 2024
- **Performance Metrics**:
  - Accuracy: ~87%
  - Balanced Accuracy: ~86%
  - F1-macro: ~0.85
- **Key Features**:
  - Uses RBF (Radial Basis Function) kernel for non-linear classification
  - Enhanced with SMOTE (Synthetic Minority Over-sampling Technique) for class balancing
  - Effective in high-dimensional spaces
- **Input Features**: D (m), H(m), Q, K(MPa)
- **Output**: Class prediction (1, 2, or 3)
- **Hyperparameters**:
  - kernel: RBF
  - C: Regularization parameter (optimized)
  - gamma: Kernel coefficient (optimized)
- **Special Notes**: 
  - Uses SMOTE preprocessing to handle class imbalance
  - Slightly better balanced accuracy compared to Random Forest
  - May require more computational resources for prediction

## Model Comparison

| Model | Accuracy | Balanced Accuracy | F1-macro | Notes |
|-------|----------|-------------------|----------|-------|
| Random Forest | ~87% | ~85% | ~0.84 | Good interpretability |
| SVM Enhanced | ~87% | ~86% | ~0.85 | Better class balancing |

## Usage

### Loading a Model
```python
import joblib
import pandas as pd

# Load the Random Forest model
rf_model = joblib.load('models/rf_tunnel_squeezing.pkl')

# Or load the SVM model
svm_model = joblib.load('models/svm_tunnel_squeezing_enhanced.pkl')
```

### Making Predictions
```python
import pandas as pd

# Prepare input data
input_data = pd.DataFrame({
    'D (m)': [6.0],
    'H(m)': [400.0],
    'Q': [1.0],
    'K(MPa)': [20.0]
})

# Make prediction
prediction = rf_model.predict(input_data)
prediction_proba = rf_model.predict_proba(input_data)

print(f"Predicted class: {prediction[0]}")
print(f"Probabilities: {prediction_proba[0]}")
```

## Model Selection Guidelines

- **Use Random Forest** when:
  - You need feature importance analysis
  - Interpretability is important
  - Fast prediction time is required

- **Use SVM Enhanced** when:
  - Class imbalance is a major concern
  - You need slightly better balanced accuracy
  - You're working with new data that might have distribution shifts

## Retraining

To retrain models with new data:
1. Update the dataset in `data/raw/tunnel.csv`
2. Run the appropriate Jupyter notebook:
   - `Tunnel_Squeezing_RandomForest.ipynb` for Random Forest
   - `Tunnel_Squeezing_SVM.ipynb` for SVM
3. Models will be saved automatically to this directory

## Model Versioning

When updating models:
- Archive previous versions with timestamp: `rf_tunnel_squeezing_YYYYMMDD.pkl`
- Update this README with new performance metrics
- Document any changes in training methodology or hyperparameters
