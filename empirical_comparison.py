#!/usr/bin/env python3
"""
Empirical Criteria Comparison for Tunnel Squeezing
===================================================

Compares machine learning predictions with traditional empirical methods:
- Singh et al. (1992) criterion
- Barla (1995) classification  
- Hoek & Maran (1998) approach
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import warnings
warnings.filterwarnings('ignore')

class EmpiricalCriteria:
    """Traditional empirical methods for tunnel squeezing assessment."""
    
    def __init__(self):
        self.gamma = 0.027  # Rock unit weight (MN/m³)
        
    def singh_criterion(self, H, Q, D):
        """
        Singh et al. (1992) squeezing criterion.
        
        Based on rock mass number N and overburden stress.
        """
        # Rock mass number N ≈ Q for practical purposes
        N = Q
        
        # Critical depth for squeezing onset
        H_critical = 275 * N**0.33 * D**0.67
        
        # Squeezing potential
        if H < H_critical:
            return 1  # Non-squeezing
        elif H < 1.5 * H_critical:
            return 2  # Minor squeezing
        else:
            return 3  # Severe squeezing
    
    def barla_classification(self, H, Q, K, D):
        """
        Barla (1995) squeezing classification.
        
        Uses strain estimation from rock mass properties.
        """
        # Rock mass deformation modulus (GPa)
        if Q > 0:
            E_rm = 25 * np.log10(Q) * (Q / (1 + Q))**0.5
            E_rm = max(E_rm, 0.1)  # Minimum value
        else:
            E_rm = 0.1
        
        # Induced stress (MPa)
        sigma_ind = self.gamma * H / 1000  # Convert to GPa
        
        # Strain estimation
        strain = (sigma_ind / E_rm) * 100  # Convert to percentage
        
        # Classification based on strain
        if strain < 1.0:
            return 1
        elif strain < 2.5:
            return 2
        else:
            return 3
    
    def hoek_maran_criterion(self, H, Q, K, D):
        """
        Hoek & Maran (1998) squeezing assessment.
        
        Uses tunnel strain criterion.
        """
        # Rock mass strength (MPa)
        if Q > 0:
            sigma_cm = 5 * Q**0.33 * np.sqrt(K / 100)
        else:
            sigma_cm = 0.1
        
        # In-situ stress (MPa)
        sigma_v = self.gamma * H
        
        # Strength-stress ratio
        if sigma_cm > 0:
            ratio = sigma_cm / sigma_v
        else:
            ratio = 0.01
        
        # Tunnel strain estimation
        strain = (1 / ratio - 1) * 100
        strain = np.clip(strain, 0.01, 20)
        
        # Classification
        if strain < 1.0:
            return 1
        elif strain < 2.5:
            return 2
        else:
            return 3

class ModelComparison:
    """Compare ML models with empirical criteria."""
    
    def __init__(self):
        self.empirical = EmpiricalCriteria()
        self.load_ml_models()
        
    def load_ml_models(self):
        """Load trained ML models."""
        try:
            self.rf_model = joblib.load('rf_tunnel_squeezing.pkl')
            self.svm_model = joblib.load('svm_tunnel_squeezing_enhanced.pkl')
            self.xgb_model = joblib.load('xgb_tunnel_squeezing.pkl')
            self.xgb_scaler = joblib.load('xgb_scaler.pkl')
            print("ML models loaded successfully")
        except Exception as e:
            print(f"Error loading models: {e}")
            self.rf_model = None
            self.svm_model = None
            self.xgb_model = None
    
    def evaluate_all_methods(self, df):
        """Evaluate all prediction methods on dataset."""
        
        results = df.copy()
        
        # Empirical criteria predictions
        results['Singh'] = results.apply(
            lambda row: self.empirical.singh_criterion(
                row['H(m)'], row['Q'], row['D (m)']
            ), axis=1
        )
        
        results['Barla'] = results.apply(
            lambda row: self.empirical.barla_classification(
                row['H(m)'], row['Q'], row['K(MPa)'], row['D (m)']
            ), axis=1
        )
        
        results['Hoek'] = results.apply(
            lambda row: self.empirical.hoek_maran_criterion(
                row['H(m)'], row['Q'], row['K(MPa)'], row['D (m)']
            ), axis=1
        )
        
        # ML model predictions
        if self.rf_model:
            X = df[['D (m)', 'H(m)', 'Q', 'K(MPa)']]
            results['RandomForest'] = self.rf_model.predict(X)
        
        if self.svm_model:
            results['SVM'] = self.svm_model.predict(X)
        
        if self.xgb_model:
            X_scaled = self.xgb_scaler.transform(X)
            xgb_pred = self.xgb_model.predict(X_scaled)
            results['XGBoost'] = xgb_pred + 1  # Convert back to 1-3 scale
        
        return results
    
    def calculate_performance_metrics(self, results_df):
        """Calculate performance metrics for all methods."""
        
        methods = ['Singh', 'Barla', 'Hoek', 'RandomForest', 'SVM', 'XGBoost']
        actual = results_df['Class']
        
        metrics = []
        
        for method in methods:
            if method in results_df.columns:
                predicted = results_df[method]
                
                accuracy = accuracy_score(actual, predicted)
                
                # Calculate per-class accuracy
                class_accuracies = {}
                for cls in [1, 2, 3]:
                    mask = actual == cls
                    if mask.sum() > 0:
                        class_acc = accuracy_score(actual[mask], predicted[mask])
                        class_accuracies[f'Class_{cls}'] = class_acc
                
                metrics.append({
                    'Method': method,
                    'Overall_Accuracy': accuracy,
                    **class_accuracies
                })
        
        return pd.DataFrame(metrics)
    
    def create_confusion_matrices(self, results_df):
        """Create confusion matrices for all methods."""
        
        methods = ['Singh', 'Barla', 'Hoek', 'RandomForest', 'SVM', 'XGBoost']
        actual = results_df['Class']
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.ravel()
        
        class_names = ['Non-squeezing', 'Minor', 'Severe']
        
        for i, method in enumerate(methods):
            if method in results_df.columns:
                predicted = results_df[method]
                cm = confusion_matrix(actual, predicted)
                
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                           xticklabels=class_names, yticklabels=class_names,
                           ax=axes[i])
                axes[i].set_title(f'{method} Method')
                axes[i].set_xlabel('Predicted')
                axes[i].set_ylabel('Actual')
            else:
                axes[i].text(0.5, 0.5, f'{method}\nNot Available', 
                           ha='center', va='center', transform=axes[i].transAxes)
                axes[i].set_title(f'{method} Method')
        
        plt.tight_layout()
        plt.savefig('empirical_confusion_matrices.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_performance_comparison(self, metrics_df):
        """Plot performance comparison across methods."""
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Overall accuracy comparison
        sns.barplot(data=metrics_df, x='Method', y='Overall_Accuracy', ax=axes[0])
        axes[0].set_title('Overall Accuracy Comparison')
        axes[0].set_ylim(0, 1)
        axes[0].tick_params(axis='x', rotation=45)
        
        # Add value labels
        for i, v in enumerate(metrics_df['Overall_Accuracy']):
            axes[0].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
        
        # Class-wise accuracy
        class_cols = [col for col in metrics_df.columns if col.startswith('Class_')]
        class_metrics = metrics_df.melt(
            id_vars=['Method'], 
            value_vars=class_cols,
            var_name='Class', 
            value_name='Accuracy'
        )
        class_metrics['Class'] = class_metrics['Class'].str.replace('Class_', 'Class ')
        
        sns.barplot(data=class_metrics, x='Method', y='Accuracy', hue='Class', ax=axes[1])
        axes[1].set_title('Class-wise Accuracy Comparison')
        axes[1].set_ylim(0, 1)
        axes[1].tick_params(axis='x', rotation=45)
        axes[1].legend(title='Squeezing Class')
        
        plt.tight_layout()
        plt.savefig('empirical_performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()

def main():
    """Main comparison analysis."""
    
    print("Empirical Criteria vs ML Models Comparison")
    print("=" * 60)
    
    # Load enhanced dataset
    try:
        df = pd.read_csv('tunnel_enhanced.csv')
        print(f"Loaded dataset: {len(df)} cases")
    except FileNotFoundError:
        print("Enhanced dataset not found!")
        return
    
    # Initialize comparison
    comparison = ModelComparison()
    
    # Evaluate all methods
    print("\nEvaluating prediction methods...")
    results = comparison.evaluate_all_methods(df)
    print(f"Evaluated {len(results.columns) - 7} prediction methods")
    
    # Calculate performance metrics
    print("\nCalculating performance metrics...")
    metrics = comparison.calculate_performance_metrics(results)
    
    # Display results
    print("\nPerformance Results:")
    print(metrics.round(4))
    
    # Create visualizations
    print("\nCreating visualizations...")
    comparison.create_confusion_matrices(results)
    comparison.plot_performance_comparison(metrics)
    
    # Save results
    results.to_csv('empirical_comparison_results.csv', index=False)
    metrics.to_csv('empirical_performance_metrics.csv', index=False)
    
    print("\nResults saved:")
    print("  - empirical_comparison_results.csv")
    print("  - empirical_performance_metrics.csv")
    print("  - empirical_confusion_matrices.png")
    print("  - empirical_performance_comparison.png")
    
    # Key findings
    print("\nKey Findings:")
    best_method = metrics.loc[metrics['Overall_Accuracy'].idxmax(), 'Method']
    best_accuracy = metrics['Overall_Accuracy'].max()
    
    print(f"  Best performing method: {best_method} ({best_accuracy:.1%} accuracy)")
    
    # Class 2 performance (most challenging)
    class_2_methods = metrics[['Method', 'Class_2']].sort_values('Class_2', ascending=False)
    best_class_2 = class_2_methods.iloc[0]
    print(f"  Best for Class 2 (minor squeezing): {best_class_2['Method']} ({best_class_2['Class_2']:.1%})")
    
    print("\nComparison analysis complete!")

if __name__ == "__main__":
    main()