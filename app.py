import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

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

# Load Models
@st.cache_resource
def load_models():
    try:
        rf_model = joblib.load('models/rf_tunnel_squeezing.pkl')
        svm_model = joblib.load('models/svm_tunnel_squeezing_enhanced.pkl')
        return rf_model, svm_model
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None

rf_model, svm_model = load_models()

# Sidebar Inputs
st.sidebar.header("🔧 Input Parameters")

def user_input_features():
    D = st.sidebar.slider("Tunnel Diameter (D)", min_value=1.0, max_value=20.0, value=6.0, step=0.1, help="Excavation diameter in meters")
    H = st.sidebar.slider("Overburden Height (H)", min_value=10.0, max_value=3000.0, value=400.0, step=10.0, help="Depth of tunnel below surface in meters")
    Q = st.sidebar.number_input("Rock Mass Quality (Q)", min_value=0.001, max_value=1000.0, value=1.0, step=0.1, format="%.3f", help="Barton's Q-value (logarithmic scale typical)")
    K = st.sidebar.slider("Rock Mass Stiffness (K)", min_value=0.0, max_value=10000.0, value=20.0, step=10.0, help="Stiffness in MPa (Note: check units of training data)")
    
    data = {'D (m)': [D],
            'H(m)': [H],
            'Q': [Q],
            'K(MPa)': [K]}
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
    model_choice = st.radio("Select Model", ["Random Forest", "SVM (Support Vector Machine)"])
    
    st.info(f"""
    **Current Input:**
    - **D**: {input_df['D (m)'][0]} m
    - **H**: {input_df['H(m)'][0]} m
    - **Q**: {input_df['Q'][0]}
    - **K**: {input_df['K(MPa)'][0]} MPa
    """)

# make prediction
if rf_model and svm_model:
    if model_choice == "Random Forest":
        model = rf_model
    else:
        model = svm_model

    prediction = model.predict(input_df)
    prediction_proba = model.predict_proba(input_df)
    
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
        
        # Explainability placeholder
        with st.expander("ℹ️ Engineering Interpretation"):
            st.markdown("""
            - **Non-squeezing**: Stable conditions, standard support.
            - **Minor**: Convergence 1-2.5%, requiring monitoring.
            - **Severe**: >2.5% strain, heavy flowing ground, requiring yielding support.
            """)

else:
    st.warning("Models not loaded. Please check .pkl files.")

st.write("---")
st.markdown("**Author:** Sudip Adhikari | **Powered by:** Streamlit & Scikit-Learn")
