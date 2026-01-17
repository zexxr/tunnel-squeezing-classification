# 🚇 Tunnel Squeezing Classification

![Python](https://img.shields.io/badge/Python-3.8+-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.2+-F7931E?logo=scikit-learn&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Complete-success)

Machine learning models to predict **tunnel squeezing severity** for geotechnical risk assessment in underground construction.

---

## 🎯 Problem Statement

**Tunnel squeezing** is time-dependent rock deformation that occurs when weak rock under high stress flows into the tunnel opening. It causes:
- TBM entrapment and equipment damage
- Lining distress and structural failure  
- Costly construction delays

This project builds ML classifiers to predict squeezing severity from geological parameters.

---

## 📊 Classification

| Class | Severity | Strain (ε) | Engineering Response |
|-------|----------|------------|---------------------|
| **1** | Non-squeezing | < 1% | Standard support |
| **2** | Minor | 1% - 2.5% | Enhanced monitoring |
| **3** | Severe | ≥ 2.5% | Heavy support required |

---

## 🔬 Features

| Feature | Description | Unit |
|---------|-------------|------|
| **D** | Tunnel diameter | m |
| **H** | Overburden depth | m |
| **Q** | Rock mass quality index | - |
| **K** | Rock mass stiffness | MPa |

---

## 🤖 Models

### 1. Support Vector Machine (SVM)
- RBF kernel with optimized C and gamma
- SMOTE for class balancing
- Extensive hyperparameter tuning

### 2. Random Forest
- Ensemble of decision trees
- Built-in feature importance
- Permutation importance analysis

---

## 📈 Performance

| Model | Accuracy | Balanced Accuracy | F1-macro |
|-------|----------|-------------------|----------|
| **SVM** | 87% | 86% | 0.85 |
| **Random Forest** | 87% | 85% | 0.84 |

---

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/zexxr/tunnel-squeezing-classification.git
cd tunnel-squeezing-classification

# Install dependencies
pip install -r requirements.txt
```

### Usage

#### 1. Python Models
```python
import joblib

# Load trained model
model = joblib.load('rf_tunnel_squeezing.pkl')

# Predict squeezing class
sample = [[6.0, 400, 0.3, 20]]  # D, H, Q, K
prediction = model.predict(sample)
print(f"Predicted Class: {prediction[0]}")
```

#### 2. 🌟 Streamlit Web App (Interactive)
Evaluate the models using a beautiful web interface:

```bash
# Run the app locally
streamlit run app.py
```
This will open a dashboard in your browser where you can adjust sliders for $D, H, Q, K$ and see real-time predictions.

---

## 📁 Project Structure

```
tunnel-squeezing-classification/
├── 📓 Tunnel_Squeezing_SVM.ipynb          # SVM analysis notebook
├── 📓 Tunnel_Squeezing_RandomForest.ipynb # Random Forest notebook
├── 📊 tunnel.csv                           # Dataset (115 samples)
├── 🤖 svm_tunnel_squeezing_enhanced.pkl   # Trained SVM model
├── 🤖 rf_tunnel_squeezing.pkl             # Trained RF model
├── 📋 requirements.txt                     # Dependencies
├── 📄 LICENSE                              # MIT License
└── 📖 README.md                            # This file
```

---

## 📊 Dataset

- **Samples**: 115 tunnel case studies
- **Source**: Published literature and field measurements
- **Class Distribution**: Imbalanced (Class 2 underrepresented)

---

## 🔍 Key Findings

1. **Most Important Features**: Overburden depth (H) and rock stiffness (K)
2. **Class 2 Challenge**: Minor squeezing is hardest to predict (transition zone)
3. **SMOTE Effectiveness**: Improves minority class recall significantly

---

## ⚠️ Limitations

- Small dataset (~115 samples)
- Class imbalance affects minority class prediction
- No site-specific geological factors included
- Static prediction (no time-dependent behavior)

---

## 🔮 Future Work

- [ ] Expand dataset with more case studies
- [ ] Add XGBoost comparison
- [ ] Compare with empirical criteria (Singh, Barla, Hoek)
- [ ] Build interactive Streamlit demo
- [ ] Add SHAP explainability analysis

---

## 📚 References

- Barla, G. (1995). Squeezing rocks in tunnels
- Hoek, E. (2001). Big tunnels in bad rock
- Singh, B. et al. (1992). Rock mass classification

---

## 👤 Author

**Sudip Adhikari** ([@zexxr](https://github.com/zexxr))

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<p align="center">
  <i>Built for tunnel engineering research and geotechnical risk assessment</i>
</p>
