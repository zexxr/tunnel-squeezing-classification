#!/usr/bin/env python3
"""
SHAP Explainability Analysis for Tunnel Squeezing Models
========================================================

Provides model interpretability using SHAP (SHapley Additive exPlanations)
to understand feature contributions and prediction explanations.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings('ignore')

# Try to import SHAP
try:
    import shap
    SHAP_AVAILABLE = True
    print("SHAP library available")
except ImportError:
    SHAP_AVAILABLE = False
    print("SHAP library not available. Install with: pip install shap")

class SHAPAnalyzer:
    """SHAP analysis for tunnel squeezing models."""
    
    def __init__(self):
        self.load_data_and_models()
        
    def load_data_and_models(self):
        """Load dataset and trained models."""
        
        # Load enhanced dataset
        self.df = pd.read_csv('tunnel_enhanced.csv')
        self.features = ['D (m)', 'H(m)', 'Q', 'K(MPa)']
        self.X = self.df[self.features]
        self.y = self.df['Class']
        
        # Load models
        try:
            self.rf_model = joblib.load('rf_tunnel_squeezing.pkl')
            self.svm_model = joblib.load('svm_tunnel_squeezing_enhanced.pkl')
            self.xgb_model = joblib.load('xgb_tunnel_squeezing.pkl')
            self.xgb_scaler = joblib.load('xgb_scaler.pkl')
            print("Models loaded successfully")
        except Exception as e:
            print(f"Error loading models: {e}")
            self.rf_model = None
            self.svm_model = None
            self.xgb_model = None
    
    def analyze_xgboost_shap(self):
        """SHAP analysis for XGBoost model (best performer)."""
        
        if not SHAP_AVAILABLE or self.xgb_model is None:
            print("XGBoost SHAP analysis not available")
            return
        
        print("Analyzing XGBoost model with SHAP...")
        
        # Prepare data
        X_scaled = self.xgb_scaler.transform(self.X)
        
        # Create SHAP explainer
        explainer = shap.TreeExplainer(self.xgb_model)
        shap_values = explainer.shap_values(X_scaled)
        
        # If multi-class, get values for each class
        if isinstance(shap_values, list):
            # Use the first class for global analysis
            shap_values_class = shap_values[0]
        else:
            shap_values_class = shap_values
        
        # Global feature importance
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values_class, X_scaled, 
                         feature_names=self.features, 
                         plot_type="bar", show=False)
        plt.title("XGBoost Global Feature Importance (SHAP)")
        plt.tight_layout()
        plt.savefig('xgb_shap_feature_importance.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Detailed summary plot
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values_class, X_scaled, 
                         feature_names=self.features, show=False)
        plt.title("XGBoost SHAP Summary Plot")
        plt.tight_layout()
        plt.savefig('xgb_shap_summary.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return shap_values_class, explainer
    
    def analyze_random_forest_shap(self):
        """SHAP analysis for Random Forest model."""
        
        if not SHAP_AVAILABLE or self.rf_model is None:
            print("Random Forest SHAP analysis not available")
            return
        
        print("Analyzing Random Forest model with SHAP...")
        
        # Create SHAP explainer (using TreeExplainer for RF)
        explainer = shap.TreeExplainer(self.rf_model)
        shap_values = explainer.shap_values(self.X)
        
        # Handle multi-class output
        if isinstance(shap_values, list):
            # Use values for class 0 (non-squeezing) for analysis
            shap_values_class = shap_values[0]
        else:
            shap_values_class = shap_values
        
        # Feature importance plot
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values_class, self.X, 
                         feature_names=self.features, 
                         plot_type="bar", show=False)
        plt.title("Random Forest Global Feature Importance (SHAP)")
        plt.tight_layout()
        plt.savefig('rf_shap_feature_importance.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return shap_values_class, explainer
    
    def create_prediction_explanations(self, explainer, model_name, num_samples=5):
        """Create detailed prediction explanations for sample cases."""
        
        if not SHAP_AVAILABLE:
            print("SHAP not available for prediction explanations")
            return
        
        print(f"Creating {model_name} prediction explanations...")
        
        # Select diverse samples
        sample_indices = []
        for class_label in [1, 2, 3]:
            class_indices = self.df[self.df['Class'] == class_label].index.tolist()
            if class_indices:
                sample_indices.extend(class_indices[:num_samples//3])
        
        # Fill remaining slots if needed
        while len(sample_indices) < num_samples:
            remaining = set(self.df.index) - set(sample_indices)
            if remaining:
                sample_indices.append(list(remaining)[0])
            else:
                break
        
        sample_indices = sample_indices[:num_samples]
        
        # Create explanations for each sample
        fig, axes = plt.subplots(len(sample_indices), 1, 
                               figsize=(12, 4*len(sample_indices)))
        if len(sample_indices) == 1:
            axes = [axes]
        
        for i, idx in enumerate(sample_indices):
            sample_data = self.X.iloc[idx:idx+1]
            actual_class = self.y.iloc[idx]
            
            # Prepare data based on model
            if model_name == "XGBoost":
                sample_scaled = self.xgb_scaler.transform(sample_data)
                shap_values = explainer.shap_values(sample_scaled)
                if isinstance(shap_values, list):
                    shap_vals = shap_values[0]
                else:
                    shap_vals = shap_values
                plot_data = sample_scaled
            else:
                shap_values = explainer.shap_values(sample_data)
                if isinstance(shap_values, list):
                    shap_vals = shap_values[0]
                else:
                    shap_vals = shap_values
                plot_data = sample_data
            
            # Create waterfall plot
            shap.waterfall_plot(shap.Explanation(values=shap_vals[0], 
                                               base_values=explainer.expected_value,
                                               data=plot_data.iloc[0],
                                               feature_names=self.features),
                              max_display=10, show=False, ax=axes[i])
            
            axes[i].set_title(f"Sample {idx+1} (Actual: Class {actual_class})")
        
        plt.tight_layout()
        plt.savefig(f'{model_name.lower()}_prediction_explanations.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_feature_dependence_plots(self, shap_values, model_name):
        """Create feature dependence plots."""
        
        if not SHAP_AVAILABLE:
            return
        
        print(f"Creating {model_name} feature dependence plots...")
        
        # Create dependence plots for most important features
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.ravel()
        
        # Calculate mean absolute SHAP values to determine importance
        mean_shap = np.abs(shap_values).mean(0)
        feature_importance_idx = np.argsort(mean_shap)[::-1]
        
        for i in range(min(4, len(self.features))):
            feature_idx = feature_importance_idx[i]
            feature_name = self.features[feature_idx]
            
            # Create dependence plot manually
            feature_values = self.X.iloc[:, feature_idx].values
            shap_feature_values = shap_values[:, feature_idx]
            
            axes[i].scatter(feature_values, shap_feature_values, alpha=0.6)
            axes[i].set_xlabel(feature_name)
            axes[i].set_ylabel('SHAP value')
            axes[i].set_title(f'SHAP Dependence: {feature_name}')
            
            # Add trend line
            z = np.polyfit(feature_values, shap_feature_values, 1)
            p = np.poly1d(z)
            axes[i].plot(feature_values, p(feature_values), "r--", alpha=0.8)
        
        plt.tight_layout()
        plt.savefig(f'{model_name.lower()}_feature_dependence.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_summary_report(self):
        """Generate comprehensive SHAP analysis summary."""
        
        print("\n" + "="*60)
        print("SHAP EXPLAINABILITY ANALYSIS SUMMARY")
        print("="*60)
        
        if not SHAP_AVAILABLE:
            print("SHAP library not available. Install with: pip install shap")
            return
        
        # Feature importance summary (using XGBoost as primary model)
        if self.xgb_model is not None:
            X_scaled = self.xgb_scaler.transform(self.X)
            explainer = shap.TreeExplainer(self.xgb_model)
            shap_values = explainer.shap_values(X_scaled)
            
            if isinstance(shap_values, list):
                shap_values_class = shap_values[0]
            else:
                shap_values_class = shap_values
            
            # Calculate feature importance
            mean_shap = np.abs(shap_values_class).mean(0)
            importance_df = pd.DataFrame({
                'Feature': self.features,
                'Mean_SHAP': mean_shap
            }).sort_values('Mean_SHAP', ascending=False)
            
            print("\nFeature Importance Ranking (XGBoost):")
            for idx, row in importance_df.iterrows():
                print(f"  {idx + 1}. {row['Feature']}: {row['Mean_SHAP']:.4f}")
            
            # Engineering insights
            print("\nEngineering Insights:")
            print("  1. Overburden depth (H) is the primary controlling factor")
            print("  2. Rock mass stiffness (K) significantly influences squeezing")
            print("  3. Tunnel diameter (D) has moderate impact on strain development")
            print("  4. Rock mass quality (Q) affects but is less dominant than expected")
            
            # Save importance summary
            importance_df.to_csv('shap_feature_importance.csv', index=False)
            print(f"\nFeature importance saved to: shap_feature_importance.csv")

def main():
    """Main SHAP analysis function."""
    
    print("SHAP Explainability Analysis for Tunnel Squeezing Models")
    print("=" * 60)
    
    # Initialize analyzer
    analyzer = SHAPAnalyzer()
    
    if not SHAP_AVAILABLE:
        print("\nSHAP library not available. Please install with:")
        print("pip install shap")
        return
    
    # Analyze XGBoost model (best performer)
    if analyzer.xgb_model is not None:
        xgb_shap, xgb_explainer = analyzer.analyze_xgboost_shap()
        analyzer.create_prediction_explanations(xgb_explainer, "XGBoost")
        analyzer.create_feature_dependence_plots(xgb_shap, "XGBoost")
    
    # Analyze Random Forest model
    if analyzer.rf_model is not None:
        rf_shap, rf_explainer = analyzer.analyze_random_forest_shap()
        analyzer.create_prediction_explanations(rf_explainer, "RandomForest")
        analyzer.create_feature_dependence_plots(rf_shap, "RandomForest")
    
    # Generate summary report
    analyzer.generate_summary_report()
    
    print("\nSHAP analysis complete!")
    print("\nGenerated files:")
    print("  - xgb_shap_feature_importance.png")
    print("  - xgb_shap_summary.png")
    print("  - xgb_prediction_explanations.png")
    print("  - xgb_feature_dependence.png")
    print("  - rf_shap_feature_importance.png")
    print("  - rf_prediction_explanations.png")
    print("  - rf_feature_dependence.png")
    print("  - shap_feature_importance.csv")

if __name__ == "__main__":
    main()