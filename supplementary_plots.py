# Supplementary Plots for Squeezing Classification Analysis
# Run this script AFTER executing the main notebook

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, learning_curve
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.calibration import calibration_curve
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-whitegrid')

# Load and prepare data
df = pd.read_csv('tunnel.csv')
df_clean = df[df['K(MPa)'] > 0].copy()

X = df_clean[['D (m)', 'H(m)', 'Q', 'K(MPa)']].values
y = df_clean['Class'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)

# Train models
svm_model = SVC(kernel='rbf', C=10, gamma='scale', probability=True, random_state=42)
svm_model.fit(X_train_smote, y_train_smote)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_smote, y_train_smote)

gb_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
gb_model.fit(X_train_smote, y_train_smote)

best_pred = rf_model.predict(X_test_scaled)

print("=" * 60)
print("SUPPLEMENTARY PLOTS FOR SQUEEZING CLASSIFICATION")
print("=" * 60)

# ========== 1. Decision Boundary (PCA) ==========
print("\n[1/8] Decision Boundary Visualization...")
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_scaled)

svm_pca = SVC(kernel='rbf', C=10, gamma='scale', random_state=42)
svm_pca.fit(X_train_pca, y_train)

h = 0.05
x_min, x_max = X_train_pca[:, 0].min() - 1, X_train_pca[:, 0].max() + 1
y_min, y_max = X_train_pca[:, 1].min() - 1, X_train_pca[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = svm_pca.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

plt.figure(figsize=(10, 7))
plt.contourf(xx, yy, Z, alpha=0.4, cmap='coolwarm')
scatter = plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=y_train, cmap='coolwarm', edgecolors='black', s=60)
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
plt.title('SVM Decision Boundaries (PCA 2D)', fontweight='bold')
plt.colorbar(scatter, label='Class')
plt.tight_layout()
plt.savefig('plot_01_decision_boundary.png', dpi=150)
plt.show()

# ========== 2. Learning Curves ==========
print("\n[2/8] Learning Curves...")
def plot_learning_curve(est, title, X, y, ax):
    sizes, train_scores, val_scores = learning_curve(est, X, y, cv=5, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 8))
    ax.fill_between(sizes, train_scores.mean(1) - train_scores.std(1), train_scores.mean(1) + train_scores.std(1), alpha=0.1)
    ax.fill_between(sizes, val_scores.mean(1) - val_scores.std(1), val_scores.mean(1) + val_scores.std(1), alpha=0.1, color='orange')
    ax.plot(sizes, train_scores.mean(1), 'o-', label='Train')
    ax.plot(sizes, val_scores.mean(1), 'o-', color='orange', label='Val')
    ax.set_title(title, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
plot_learning_curve(SVC(kernel='rbf', C=10), 'SVM', X_train_scaled, y_train, axes[0])
plot_learning_curve(RandomForestClassifier(n_estimators=50), 'Random Forest', X_train_scaled, y_train, axes[1])
plot_learning_curve(GradientBoostingClassifier(n_estimators=50), 'Gradient Boosting', X_train_scaled, y_train, axes[2])
plt.suptitle('Learning Curves', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('plot_02_learning_curves.png', dpi=150)
plt.show()

# ========== 3. Precision-Recall Curves ==========
print("\n[3/8] Precision-Recall Curves...")
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
colors = ['#2ecc71', '#f39c12', '#e74c3c']
names = ['Non-squeezing', 'Minor', 'Severe']

for ax, model, title in zip(axes, [svm_model, rf_model, gb_model], ['SVM', 'RF', 'GB']):
    y_score = model.predict_proba(X_test_scaled)
    y_bin = label_binarize(y_test, classes=[1,2,3])
    for i in range(3):
        prec, rec, _ = precision_recall_curve(y_bin[:,i], y_score[:,i])
        ap = average_precision_score(y_bin[:,i], y_score[:,i])
        ax.plot(rec, prec, color=colors[i], label=f'{names[i]} (AP={ap:.2f})')
    ax.set_title(title, fontweight='bold')
    ax.legend(loc='lower left', fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')

plt.suptitle('Precision-Recall Curves', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('plot_03_precision_recall.png', dpi=150)
plt.show()

# ========== 4. Calibration Curves ==========
print("\n[4/8] Calibration Curves...")
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for ax, model, title in zip(axes, [svm_model, rf_model, gb_model], ['SVM', 'RF', 'GB']):
    y_prob = model.predict_proba(X_test_scaled)[:, 2]
    y_bin = (y_test == 3).astype(int)
    prob_true, prob_pred = calibration_curve(y_bin, y_prob, n_bins=5)
    ax.plot(prob_pred, prob_true, 's-', label=title)
    ax.plot([0,1], [0,1], 'k--', label='Perfect')
    ax.set_title(f'{title} - Severe Class', fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

plt.suptitle('Calibration Curves', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('plot_04_calibration.png', dpi=150)
plt.show()

# ========== 5. SHAP (if available) ==========
print("\n[5/8] SHAP Analysis...")
try:
    import shap
    explainer = shap.TreeExplainer(rf_model)
    shap_vals = explainer.shap_values(X_test_scaled)
    X_df = pd.DataFrame(X_test_scaled, columns=['D', 'H', 'Q', 'K'])
    plt.figure(figsize=(8, 5))
    shap.summary_plot(shap_vals, X_df, plot_type='bar', show=False)
    plt.title('SHAP Feature Importance', fontweight='bold')
    plt.tight_layout()
    plt.savefig('plot_05_shap.png', dpi=150)
    plt.show()
except:
    print("  SHAP not available. Skipping...")

# ========== 6. 3D Scatter ==========
print("\n[6/8] 3D Scatter Plot...")
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
cmap = {1: '#2ecc71', 2: '#f39c12', 3: '#e74c3c'}
for c in [1,2,3]:
    m = df_clean['Class'] == c
    ax.scatter(df_clean.loc[m,'H(m)'], df_clean.loc[m,'Q'], df_clean.loc[m,'K(MPa)'], c=cmap[c], label=f'Class {c}', s=40)
ax.set_xlabel('H (m)')
ax.set_ylabel('Q')
ax.set_zlabel('K (MPa)')
ax.legend()
plt.title('3D Feature Space', fontweight='bold')
plt.tight_layout()
plt.savefig('plot_06_3d_scatter.png', dpi=150)
plt.show()

# ========== 7. CV Box Plot ==========
print("\n[7/8] CV Score Distribution...")
cv = StratifiedKFold(5, shuffle=True, random_state=42)
cv_data = pd.DataFrame({
    'SVM': cross_val_score(SVC(kernel='rbf', C=10), X_train_smote, y_train_smote, cv=cv, scoring='f1_macro'),
    'RF': cross_val_score(RandomForestClassifier(n_estimators=100), X_train_smote, y_train_smote, cv=cv, scoring='f1_macro'),
    'GB': cross_val_score(GradientBoostingClassifier(n_estimators=100), X_train_smote, y_train_smote, cv=cv, scoring='f1_macro')
})
plt.figure(figsize=(8, 5))
cv_data.boxplot(patch_artist=True)
plt.ylabel('F1 Macro')
plt.title('CV Score Distribution', fontweight='bold')
plt.tight_layout()
plt.savefig('plot_07_cv_boxplot.png', dpi=150)
plt.show()
print(cv_data.describe().loc[['mean','std']])

# ========== 8. Misclassification Analysis ==========
print("\n[8/8] Misclassification Analysis...")
mis = best_pred != y_test
test_df = pd.DataFrame(X_test, columns=['D','H','Q','K'])
test_df['Actual'] = y_test
test_df['Pred'] = best_pred

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
for ax, col in zip(axes.flat, ['D','H','Q','K']):
    ax.scatter(test_df.loc[~mis, col], test_df.loc[~mis, 'Actual'], c='#2ecc71', label='Correct', s=60)
    ax.scatter(test_df.loc[mis, col], test_df.loc[mis, 'Actual'], c='#e74c3c', marker='X', s=100, label='Misclassified')
    ax.set_xlabel(col)
    ax.set_ylabel('Actual Class')
    ax.legend()
    ax.grid(alpha=0.3)
plt.suptitle('Misclassification Analysis', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('plot_08_misclassification.png', dpi=150)
plt.show()

if mis.sum() > 0:
    print("\nMisclassified Samples:")
    print(test_df[mis])

print("\n" + "=" * 60)
print("ALL PLOTS GENERATED AND SAVED!")
print("=" * 60)
