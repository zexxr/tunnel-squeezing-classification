import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Page Config
st.set_page_config(
    page_title="Advanced Tunnel Squeezing Predictor",
    page_icon="🚇",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced styling
st.markdown("""
<style>
    .reportview-container {
        background: #f0f2f6
    }
    .big-font {
        font-size:30px !important;
        font-weight: bold;
    }
    .stAlert {
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Load Models
@st.cache_resource
def load_models():
    """Load all trained models."""
    try:
        rf_model = joblib.load('rf_tunnel_squeezing.pkl')
        svm_model = joblib.load('svm_tunnel_squeezing_enhanced.pkl')
        xgb_model = joblib.load('xgb_tunnel_squeezing.pkl')
        xgb_scaler = joblib.load('xgb_scaler.pkl')
        
        # Load feature importance data
        try:
            shap_importance = pd.read_csv('shap_feature_importance.csv')
        except:
            shap_importance = None
            
        return rf_model, svm_model, xgb_model, xgb_scaler, shap_importance
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None, None, None

rf_model, svm_model, xgb_model, xgb_scaler, shap_importance = load_models()

# Sidebar Inputs
st.sidebar.header("🔧 Tunnel Parameters")

def user_input_features():
    """Get user input parameters."""
    D = st.sidebar.slider(
        "Tunnel Diameter (D)", 
        min_value=1.0, max_value=20.0, value=6.0, step=0.1,
        help="Excavation diameter in meters"
    )
    
    H = st.sidebar.slider(
        "Overburden Depth (H)", 
        min_value=10.0, max_value=3000.0, value=400.0, step=10.0,
        help="Depth of tunnel below surface in meters"
    )
    
    Q = st.sidebar.number_input(
        "Rock Mass Quality (Q)", 
        min_value=0.001, max_value=1000.0, value=1.0, step=0.1, format="%.3f",
        help="Barton's Q-value (logarithmic scale typical)"
    )
    
    K = st.sidebar.slider(
        "Rock Mass Stiffness (K)", 
        min_value=0.0, max_value=10000.0, value=20.0, step=10.0,
        help="Stiffness in MPa"
    )
    
    data = {
        'D (m)': [D],
        'H(m)': [H],
        'Q': [Q],
        'K(MPa)': [K]
    }
    features = pd.DataFrame(data)
    return features

input_df = user_input_features()

# Enhanced Model Selection
st.sidebar.header("🤖 Model Options")
model_options = ["XGBoost (Best Performance)", "Random Forest", "SVM", "Ensemble (All Models)"]
model_choice = st.sidebar.selectbox("Select Prediction Method", model_options)

show_probabilities = st.sidebar.checkbox("Show Probability Distribution", value=True)
show_explanation = st.sidebar.checkbox("Show Feature Explanation", value=True)

# Main Panel
st.title("🚇 Advanced Tunnel Squeezing Prediction")
st.markdown("### Machine Learning based Geotechnical Risk Assessment with Multiple Models")

st.write("---")

# Model Performance Overview (if data available)
if shap_importance is not None:
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h4>XGBoost</h4>
            <p style="color: green; font-size: 20px; font-weight: bold;">98.1% Accuracy</p>
            <small>Best overall performance</small>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h4>Random Forest</h4>
            <p style="color: blue; font-size: 20px; font-weight: bold;">87.0% Accuracy</p>
            <small>Good interpretability</small>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h4>SVM</h4>
            <p style="color: orange; font-size: 20px; font-weight: bold;">87.0% Accuracy</p>
            <small>Robust classifier</small>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metric-card">
            <h4>Dataset</h4>
            <p style="color: purple; font-size: 20px; font-weight: bold;">157 Cases</p>
            <small>Enhanced with literature</small>
        </div>
        """, unsafe_allow_html=True)
    
    st.write("---")

# Input Summary
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("📊 Input Parameters")
    st.info(f"""
    **Current Tunnel Configuration:**
    - **Diameter**: {input_df['D (m)'][0]} m
    - **Overburden**: {input_df['H(m)'][0]} m
    - **Q-value**: {input_df['Q'][0]}
    - **Stiffness**: {input_df['K(MPa)'][0]} MPa
    """)
    
    # Feature importance display
    if shap_importance is not None and show_explanation:
        st.subheader("🔍 Feature Importance")
        fig, ax = plt.subplots(figsize=(6, 3))
        sns.barplot(data=shap_importance, x='Mean_SHAP', y='Feature', ax=ax)
        ax.set_title('SHAP Feature Importance (XGBoost)')
        ax.set_xlabel('Importance Score')
        st.pyplot(fig, use_container_width=True)

# Predictions
if rf_model and svm_model and xgb_model:
    
    def get_model_prediction(model_name):
        """Get prediction from specific model."""
        if model_name == "XGBoost":
            input_scaled = xgb_scaler.transform(input_df)
            prediction = xgb_model.predict(input_scaled)[0] + 1  # Convert back to 1-3 scale
            prediction_proba = xgb_model.predict_proba(input_scaled)[0]
        elif model_name == "Random Forest":
            prediction = rf_model.predict(input_df)[0]
            prediction_proba = rf_model.predict_proba(input_df)[0]
        elif model_name == "SVM":
            prediction = svm_model.predict(input_df)[0]
            prediction_proba = svm_model.predict_proba(input_df)[0]
        else:
            return None, None
        return prediction, prediction_proba
    
    # Get predictions based on model choice
    with col2:
        st.subheader("🎯 Prediction Results")
        
        if model_choice == "Ensemble (All Models)":
            # Get all model predictions
            models = ["XGBoost", "Random Forest", "SVM"]
            predictions = []
            probabilities = []
            
            for model in models:
                pred, prob = get_model_prediction(model)
                predictions.append(pred)
                probabilities.append(prob)
            
            # Ensemble prediction (majority vote)
            ensemble_pred = max(set(predictions), key=predictions.count)
            # Average probabilities
            ensemble_proba = np.mean(probabilities, axis=0)
            
            prediction = ensemble_pred
            prediction_proba = ensemble_proba
            
            # Show individual model results
            st.subheader("📈 Individual Model Results")
            model_results_df = pd.DataFrame({
                'Model': models,
                'Prediction': predictions,
                'Confidence': [prob[pred-1] for pred, prob in zip(predictions, probabilities)]
            })
            st.dataframe(model_results_df, use_container_width=True)
            
            st.markdown("---")
            st.subheader("🏆 Ensemble Prediction")
            
        else:
            # Single model prediction
            model_name = model_choice.split(' ')[0]
            prediction, prediction_proba = get_model_prediction(model_name)
        
        # Display main prediction
        class_map = {1: "Non-squeezing", 2: "Minor Squeezing", 3: "Severe Squeezing"}
        predicted_class = class_map.get(prediction, f"Class {prediction}")
        
        # Color coding result
        if prediction == 1:
            st.success(f"### Result: {predicted_class}")
        elif prediction == 2:
            st.warning(f"### Result: {predicted_class}")
        else:
            st.error(f"### Result: {predicted_class}")
            
        st.markdown(f"**Confidence:** {np.max(prediction_proba)*100:.1f}%")
        
        # Probability Distribution
        if show_probabilities:
            st.subheader("📊 Probability Distribution")
            proba_df = pd.DataFrame(
                prediction_proba.reshape(1, -1), 
                columns=['Non-squeezing', 'Minor', 'Severe']
            )
            st.bar_chart(proba_df.T)
        
        # Engineering Recommendations
        st.subheader("⚙️ Engineering Recommendations")
        
        if prediction == 1:
            st.markdown("""
            - **Support Type**: Standard support systems adequate
            - **Monitoring**: Regular convergence monitoring
            - **Construction**: Normal excavation procedures
            - **Risk Level**: Low
            """)
        elif prediction == 2:
            st.markdown("""
            - **Support Type**: Enhanced support with rock bolts
            - **Monitoring**: Intensive convergence monitoring required
            - **Construction**: Controlled excavation, possibly sequential
            - **Risk Level**: Moderate
            """)
        else:
            st.markdown("""
            - **Support Type**: Heavy yielding support systems
            - **Monitoring**: Continuous real-time monitoring
            - **Construction**: Very careful excavation, possibly forepoling
            - **Risk Level**: High - Consider route realignment
            """)

else:
    st.warning("Models not loaded. Please check .pkl files.")

# Advanced Analysis Section
st.write("---")
st.header("📚 Advanced Analysis & Insights")

tab1, tab2, tab3 = st.tabs(["Dataset Overview", "Model Comparison", "Engineering Guidelines"])

with tab1:
    st.subheader("📊 Enhanced Dataset Statistics")
    
    # Load and display dataset info
    try:
        df = pd.read_csv('tunnel_enhanced.csv')
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Class Distribution:**")
            class_counts = df['Class'].value_counts().sort_index()
            class_names = {1: "Non-squeezing", 2: "Minor", 3: "Severe"}
            for cls, count in class_counts.items():
                st.write(f"  {class_names[cls]}: {count} cases")
        
        with col2:
            st.markdown("**Parameter Ranges:**")
            st.write(f"  Diameter: {df['D (m)'].min():.1f} - {df['D (m)'].max():.1f} m")
            st.write(f"  Depth: {df['H(m)'].min():.0f} - {df['H(m)'].max():.0f} m")
            st.write(f"  Q-value: {df['Q'].min():.3f} - {df['Q'].max():.1f}")
            st.write(f"  Stiffness: {df['K(MPa)'].min():.1f} - {df['K(MPa)'].max():.0f} MPa")
        
        # Distribution plot
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        axes = axes.ravel()
        
        features = ['D (m)', 'H(m)', 'Q', 'K(MPa)']
        for i, feature in enumerate(features):
            axes[i].hist(df[feature], bins=20, alpha=0.7, edgecolor='black')
            axes[i].set_title(f'{feature} Distribution')
            axes[i].set_xlabel(feature)
            axes[i].set_ylabel('Frequency')
        
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Could not load dataset: {e}")

with tab2:
    st.subheader("🏆 Model Performance Comparison")
    
    # Load comparison results if available
    try:
        comparison_df = pd.read_csv('empirical_comparison_results.csv')
        
        # Show model performance table
        performance_data = {
            'Model': ['XGBoost', 'Random Forest', 'SVM', 'Singh Criterion', 'Barla Method'],
            'Accuracy': [0.981, 0.803, 0.771, 0.440, 0.548],
            'Strength': ['Excellent', 'Good', 'Good', 'Poor', 'Fair']
        }
        
        perf_df = pd.DataFrame(performance_data)
        st.dataframe(perf_df, use_container_width=True)
        
        st.markdown("""
        **Key Insights:**
        - **XGBoost** significantly outperforms traditional empirical methods
        - **Machine Learning** models capture non-linear interactions better
        - **Class 2 (Minor squeezing)** remains challenging for all methods
        - **Empirical methods** serve as good engineering baselines
        """)
        
    except Exception as e:
        st.warning("Detailed comparison results not available")

with tab3:
    st.subheader("📖 Engineering Guidelines")
    
    st.markdown("""
    ### Tunnel Squeezing Classification System
    
    | Class | Severity | Strain (ε) | Engineering Response |
    |-------|----------|------------|---------------------|
    | **1** | Non-squeezing | < 1% | Standard support, regular monitoring |
    | **2** | Minor | 1% - 2.5% | Enhanced support, intensive monitoring |
    | **3** | Severe | ≥ 2.5% | Heavy yielding support, risk mitigation |
    
    ### Key Factors Influencing Squeezing
    
    **Most Critical Factors (SHAP Analysis):**
    1. **Rock Mass Stiffness (K)** - Primary controlling factor
    2. **Overburden Depth (H)** - Stress-induced deformation
    3. **Rock Quality (Q)** - Mass strength and deformation
    4. **Tunnel Diameter (D)** - Opening size effects
    
    ### Practical Recommendations
    
    **For High-Risk Projects:**
    - Use multiple prediction methods
    - Incorporate site-specific geological investigations
    - Implement real-time monitoring systems
    - Consider alternative tunnel alignments
    
    **For Medium-Risk Projects:**
    - Follow standard rock mass classification systems
    - Design conservative support systems
    - Plan for potential modifications
    
    **For Low-Risk Projects:**
    - Standard design procedures adequate
    - Regular monitoring and maintenance
    """)

st.write("---")
st.markdown("""
**About This Tool**: Advanced ML-based tunnel squeezing prediction using ensemble methods.
Developed with 157+ case studies and validated against empirical criteria.
**Author**: Sudip Adhikari | **Models**: XGBoost, Random Forest, SVM
""")