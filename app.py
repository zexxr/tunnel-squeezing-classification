import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.special import softmax

# Page Config
st.set_page_config(
    page_title="Tunnel Squeezing Predictor",
    page_icon="🚇",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
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
</style>
""", unsafe_allow_html=True)

# Get base directory for robust path resolution
BASE_DIR = Path(__file__).parent

# Load Models with graceful degradation
@st.cache_resource
def load_models():
    """Load models with individual error handling for graceful degradation."""
    models = {'rf': None, 'svm': None}
    errors = []
    
    # Load Random Forest
    rf_path = BASE_DIR / 'rf_tunnel_squeezing.pkl'
    try:
        models['rf'] = joblib.load(rf_path)
    except FileNotFoundError:
        errors.append(f"Random Forest model not found: {rf_path.name}")
    except Exception as e:
        errors.append(f"Error loading Random Forest: {e}")
    
    # Load SVM
    svm_path = BASE_DIR / 'svm_tunnel_squeezing_enhanced.pkl'
    try:
        models['svm'] = joblib.load(svm_path)
    except FileNotFoundError:
        errors.append(f"SVM model not found: {svm_path.name}")
    except Exception as e:
        errors.append(f"Error loading SVM: {e}")
    
    return models, errors

models, load_errors = load_models()
rf_model = models['rf']
svm_model = models['svm']

# Display loading errors if any
if load_errors:
    for error in load_errors:
        st.error(f"⚠️ {error}")

# Sidebar Inputs
st.sidebar.header("🔧 Input Parameters")

def user_input_features():
    D = st.sidebar.slider(
        "Tunnel Diameter (D)", 
        min_value=1.0, max_value=20.0, value=6.0, step=0.1, 
        help="Excavation diameter in meters"
    )
    H = st.sidebar.slider(
        "Overburden Height (H)", 
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
        min_value=0.0, max_value=100.0, value=20.0, step=1.0, 
        help="Stiffness in GPa (typical range: 1-100 GPa)"
    )
    
    # Input validation warnings
    if D < 3.0 or D > 14.0:
        st.sidebar.warning("⚠️ Unusual diameter - typical tunnels: 3-14m")
    if H > 2000:
        st.sidebar.warning("⚠️ Very deep overburden - verify stress estimates")
    if Q < 0.01 or Q > 100:
        st.sidebar.warning("⚠️ Q-value outside typical range (0.01-100)")
    if K < 1 or K > 80:
        st.sidebar.warning("⚠️ K outside typical range (1-80 GPa)")
    
    # Note: Column names must match training data exactly
    data = {'D (m)': [D],
            'H(m)': [H],
            'Q': [Q],
            'K(MPa)': [K]}  # Keep column name for model compatibility
    features = pd.DataFrame(data)
    return features

input_df = user_input_features()

# Main Panel
st.title("🚇 Tunnel Squeezing Prediction")
st.markdown("### Machine Learning based Geotechnical Risk Assessment")

st.write("---")

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Model Settings")
    
    # Build available model list dynamically
    available_models = []
    if rf_model is not None:
        available_models.append("Random Forest")
    if svm_model is not None:
        available_models.append("SVM (Support Vector Machine)")
    
    if not available_models:
        st.error("❌ No models available. Please ensure .pkl files are present.")
        st.stop()
    
    model_choice = st.radio("Select Model", available_models)
    
    st.info(f"""
    **Current Input:**
    - **D**: {input_df['D (m)'][0]} m
    - **H**: {input_df['H(m)'][0]} m
    - **Q**: {input_df['Q'][0]}
    - **K**: {input_df['K(MPa)'][0]} GPa
    """)

# Safe probability prediction with fallback
def get_prediction_proba(model, input_data):
    """Get prediction probabilities with fallback for models without predict_proba."""
    try:
        return model.predict_proba(input_data)
    except AttributeError:
        # SVM trained without probability=True - use decision function
        try:
            decision = model.decision_function(input_data)
            # Convert decision function to pseudo-probabilities via softmax
            if decision.ndim == 1:
                decision = decision.reshape(1, -1)
            return softmax(decision, axis=1)
        except Exception:
            # Last resort: return uniform distribution
            n_classes = 3
            return np.ones((1, n_classes)) / n_classes

# Make prediction
if model_choice == "Random Forest":
    model = rf_model
else:
    model = svm_model

prediction = model.predict(input_df)
prediction_proba = get_prediction_proba(model, input_df)

class_map = {1: "Non-squeezing", 2: "Minor Squeezing", 3: "Severe Squeezing"}
predicted_class = class_map.get(prediction[0], f"Class {prediction[0]}")

# Probability Data
proba_df = pd.DataFrame(prediction_proba, columns=['Non-squeezing', 'Minor', 'Severe'])

with col2:
    st.subheader("🎯 Prediction")
    
    # Color coding result
    if prediction[0] == 1:
        st.success(f"### Result: {predicted_class}")
    elif prediction[0] == 2:
        st.warning(f"### Result: {predicted_class}")
    else:
        st.error(f"### Result: {predicted_class}")
        
    st.markdown(f"**Confidence:** {np.max(prediction_proba)*100:.1f}%")
    
    st.subheader("📊 Probability Distribution")
    st.bar_chart(proba_df.T)
    
    # SHAP Explainability
    with st.expander("ℹ️ Feature Contributions (SHAP Analysis)"):
        try:
            import shap
            
            # Use TreeExplainer for Random Forest (fast)
            if model_choice == "Random Forest" and rf_model is not None:
                explainer = shap.TreeExplainer(rf_model)
                shap_values = explainer.shap_values(input_df)
                
                # Get SHAP values for predicted class
                pred_class_idx = int(prediction[0]) - 1
                if isinstance(shap_values, list):
                    sv = shap_values[pred_class_idx][0]
                else:
                    sv = shap_values[0]
                
                # Create feature contribution display
                feature_names = ['D (m)', 'H(m)', 'Q', 'K(GPa)']
                contributions = pd.DataFrame({
                    'Feature': feature_names,
                    'Value': input_df.values[0],
                    'SHAP Contribution': sv
                }).sort_values('SHAP Contribution', key=abs, ascending=False)
                
                st.markdown("**Feature Contributions to Prediction:**")
                for _, row in contributions.iterrows():
                    direction = "↑" if row['SHAP Contribution'] > 0 else "↓"
                    st.write(f"- **{row['Feature']}** = {row['Value']:.2f}: {direction} {abs(row['SHAP Contribution']):.3f}")
            else:
                # Fallback for SVM - show general interpretation
                st.markdown("""
                **General Interpretation:**
                - **Non-squeezing**: Stable conditions, standard support.
                - **Minor**: Convergence 1-2.5%, requiring monitoring.
                - **Severe**: >2.5% strain, heavy flowing ground, requiring yielding support.
                
                *Note: SHAP analysis optimized for Random Forest model.*
                """)
        except ImportError:
            st.markdown("""
            **Engineering Interpretation:**
            - **Non-squeezing**: Stable conditions, standard support.
            - **Minor**: Convergence 1-2.5%, requiring monitoring.
            - **Severe**: >2.5% strain, heavy flowing ground, requiring yielding support.
            
            *Install `shap` package for detailed feature explanations.*
            """)
        except Exception as e:
            st.markdown(f"""
            **Engineering Interpretation:**
            - **Non-squeezing**: Stable conditions, standard support.
            - **Minor**: Convergence 1-2.5%, requiring monitoring.
            - **Severe**: >2.5% strain, heavy flowing ground, requiring yielding support.
            """)

st.write("---")
st.markdown("**Author:** Sudip Adhikari | **Powered by:** Streamlit & Scikit-Learn")
