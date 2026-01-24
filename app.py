import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from io import BytesIO

# Page Config
st.set_page_config(
    page_title="Tunnel Squeezing Predictor",
    page_icon="🚇",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
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
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Load Models with error handling
@st.cache_resource
def load_models():
    """Load both RF and SVM models with error handling"""
    rf_model = None
    svm_model = None
    errors = []
    
    try:
        rf_model = joblib.load('rf_tunnel_squeezing.pkl')
    except Exception as e:
        errors.append(f"RF Model: {str(e)}")
    
    try:
        svm_model = joblib.load('svm_tunnel_squeezing_enhanced.pkl')
    except Exception as e:
        errors.append(f"SVM Model: {str(e)}")
    
    if errors:
        for error in errors:
            st.sidebar.error(f"Error loading model: {error}")
    
    return rf_model, svm_model

# Input validation function
def validate_inputs(D: float, H: float, Q: float, K: float) -> dict:
    """
    Validate input parameters and return warnings.
    
    Returns:
        dict with 'is_valid', 'warnings', and 'status' keys
    """
    warnings = []
    status = {}
    
    # Check D (Tunnel diameter)
    if D < 2 or D > 20:
        warnings.append(f"⚠️ Tunnel diameter ({D}m) is outside typical range (2-20m)")
        status['D'] = '⚠️'
    else:
        status['D'] = '✅'
    
    # Check H (Overburden depth)
    if H < 10 or H > 2000:
        warnings.append(f"⚠️ Overburden depth ({H}m) is outside typical range (10-2000m)")
        status['H'] = '⚠️'
    else:
        status['H'] = '✅'
    
    # Check Q (Rock quality)
    if Q < 0.001 or Q > 1000:
        warnings.append(f"⚠️ Rock quality Q ({Q}) is outside typical range (0.001-1000)")
        status['Q'] = '⚠️'
    else:
        status['Q'] = '✅'
    
    # Check K (Rock stiffness) - Note: units are MPa
    if K < 0.1 or K > 10000:  # Keeping upper bound flexible
        warnings.append(f"⚠️ Rock stiffness ({K} MPa) may be outside typical range (0.1-10000 MPa)")
        status['K'] = '⚠️'
    else:
        status['K'] = '✅'
    
    # Additional engineering warnings
    if H / D > 500:
        warnings.append(f"⚠️ Very high H/D ratio ({H/D:.1f}) - extreme depth conditions")
    
    if Q < 0.01 and K > 1000:
        warnings.append(f"⚠️ Unusual combination: very poor rock quality with high stiffness")
    
    return {
        'is_valid': len(warnings) == 0,
        'warnings': warnings,
        'status': status
    }

# Prediction with confidence function
def get_prediction_with_confidence(model, features, model_name="Model"):
    """
    Get prediction with probability distribution.
    
    Returns:
        dict with 'class', 'confidence', 'probabilities', 'class_name'
    """
    try:
        prediction = model.predict(features)[0]
        probabilities = model.predict_proba(features)[0]
        
        # Map prediction to class (assuming 1-indexed classes)
        class_map = {1: "Non-squeezing", 2: "Minor Squeezing", 3: "Severe Squeezing"}
        class_name = class_map.get(prediction, f"Class {prediction}")
        
        # Get confidence for predicted class (assuming classes are 1, 2, 3)
        confidence = probabilities[prediction - 1] * 100
        
        return {
            'class': prediction,
            'class_name': class_name,
            'confidence': confidence,
            'probabilities': probabilities,
            'success': True,
            'error': None
        }
    except Exception as e:
        return {
            'class': None,
            'class_name': None,
            'confidence': 0,
            'probabilities': None,
            'success': False,
            'error': str(e)
        }

# Visualization functions
def create_confidence_gauge(confidence, class_name):
    """Create a gauge chart for confidence score"""
    # Determine color based on confidence level
    if confidence >= 80:
        color = "green"
    elif confidence >= 60:
        color = "yellow"
    else:
        color = "red"
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = confidence,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': f"Confidence: {class_name}", 'font': {'size': 16}},
        number = {'suffix': "%", 'font': {'size': 30}},
        gauge = {
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': color},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 60], 'color': '#ffcccc'},
                {'range': [60, 80], 'color': '#ffffcc'},
                {'range': [80, 100], 'color': '#ccffcc'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=20))
    return fig

def create_probability_bars(probabilities):
    """Create horizontal bar chart for probability distribution"""
    class_names = ['Non-squeezing', 'Minor Squeezing', 'Severe Squeezing']
    colors = ['#28a745', '#ffc107', '#dc3545']
    
    fig = go.Figure(data=[
        go.Bar(
            y=class_names,
            x=probabilities * 100,
            orientation='h',
            marker=dict(
                color=colors,
                line=dict(color='rgb(8,48,107)', width=1.5)
            ),
            text=[f'{p*100:.1f}%' for p in probabilities],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title="Probability Distribution Across Classes",
        xaxis_title="Probability (%)",
        yaxis_title="Squeezing Class",
        height=300,
        margin=dict(l=20, r=20, t=40, b=20),
        showlegend=False
    )
    
    fig.update_xaxes(range=[0, 100])
    
    return fig

def create_feature_ranges_plot(D, H, Q, K):
    """Create a visualization showing where inputs fall in typical ranges"""
    features_data = {
        'Feature': ['D (Tunnel Diameter)', 'H (Overburden Depth)', 'Q (Rock Quality)', 'K (Rock Stiffness)'],
        'Value': [D, H, Q, K],
        'Min': [2, 10, 0.001, 0.1],
        'Max': [20, 2000, 1000, 10000],
        'Unit': ['m', 'm', '-', 'MPa']
    }
    
    df = pd.DataFrame(features_data)
    
    # Calculate percentage within range (using log scale for Q)
    percentages = []
    for i, row in df.iterrows():
        if row['Feature'] == 'Q (Rock Quality)':
            # Use log scale for Q
            log_val = np.log10(row['Value'])
            log_min = np.log10(row['Min'])
            log_max = np.log10(row['Max'])
            pct = ((log_val - log_min) / (log_max - log_min)) * 100
        else:
            pct = ((row['Value'] - row['Min']) / (row['Max'] - row['Min'])) * 100
        percentages.append(min(max(pct, 0), 100))  # Clamp to 0-100
    
    df['Percentage'] = percentages
    
    fig = go.Figure()
    
    # Add bars showing position within range
    fig.add_trace(go.Bar(
        y=df['Feature'],
        x=df['Percentage'],
        orientation='h',
        marker=dict(color='lightblue'),
        text=[f"{val} {unit}" for val, unit in zip(df['Value'], df['Unit'])],
        textposition='auto',
        name='Current Value'
    ))
    
    fig.update_layout(
        title="Input Values Position Within Typical Ranges",
        xaxis_title="Position in Range (%)",
        yaxis_title="",
        height=300,
        margin=dict(l=20, r=20, t=40, b=20),
        showlegend=False
    )
    
    fig.update_xaxes(range=[0, 100])
    
    return fig

# Example scenarios
EXAMPLE_SCENARIOS = {
    "Custom": None,
    "Non-Squeezing (Stable)": {
        'D': 6.0,
        'H': 150.0,
        'Q': 5.0,
        'K': 1000.0,
        'description': "Competent rock, moderate depth, minimal support required"
    },
    "Minor Squeezing": {
        'D': 8.0,
        'H': 500.0,
        'Q': 0.5,
        'K': 50.0,
        'description': "Weak rock, significant depth, enhanced support needed"
    },
    "Severe Squeezing": {
        'D': 10.0,
        'H': 800.0,
        'Q': 0.05,
        'K': 10.0,
        'description': "Very weak rock, great depth, heavy deformable support required"
    }
}

# Load models
rf_model, svm_model = load_models()

# Sidebar for inputs
with st.sidebar:
    st.header("🔧 Input Parameters")
    
    # Example scenario selector
    scenario = st.selectbox(
        "Quick Examples",
        list(EXAMPLE_SCENARIOS.keys()),
        help="Select a preset scenario or choose 'Custom' to enter your own values"
    )
    
    # Set default values based on scenario
    if scenario != "Custom" and EXAMPLE_SCENARIOS[scenario] is not None:
        default_values = EXAMPLE_SCENARIOS[scenario]
        st.info(f"📋 **Scenario:** {default_values['description']}")
        D_default = default_values['D']
        H_default = default_values['H']
        Q_default = default_values['Q']
        K_default = default_values['K']
    else:
        D_default = 6.0
        H_default = 400.0
        Q_default = 1.0
        K_default = 20.0
    
    st.markdown("---")
    
    # Input sliders with validation indicators
    D = st.slider(
        "Tunnel Diameter (D)",
        min_value=1.0,
        max_value=25.0,
        value=D_default,
        step=0.1,
        help="Excavation diameter in meters. Typical range: 2-20m"
    )
    
    H = st.slider(
        "Overburden Depth (H)",
        min_value=5.0,
        max_value=3000.0,
        value=H_default,
        step=10.0,
        help="Depth of tunnel below surface in meters. Typical range: 10-2000m"
    )
    
    Q = st.number_input(
        "Rock Mass Quality (Q)",
        min_value=0.001,
        max_value=1000.0,
        value=Q_default,
        step=0.1,
        format="%.3f",
        help="Barton's Q-value (logarithmic scale). Typical range: 0.001-1000"
    )
    
    K = st.slider(
        "Rock Mass Stiffness (K)",
        min_value=0.1,
        max_value=10000.0,
        value=K_default,
        step=10.0,
        help="Deformation modulus in MPa. Typical range: 0.1-10000 MPa"
    )
    
    st.markdown("---")
    
    # Validate inputs
    validation = validate_inputs(D, H, Q, K)
    
    # Display validation status
    st.subheader("Input Validation")
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"{validation['status'].get('D', '❓')} Diameter")
        st.write(f"{validation['status'].get('H', '❓')} Depth")
    with col2:
        st.write(f"{validation['status'].get('Q', '❓')} Quality")
        st.write(f"{validation['status'].get('K', '❓')} Stiffness")
    
    # Display warnings if any
    if validation['warnings']:
        st.warning("⚠️ **Warnings:**")
        for warning in validation['warnings']:
            st.write(warning)
    else:
        st.success("✅ All inputs within normal range")

# Prepare input dataframe
input_df = pd.DataFrame({
    'D (m)': [D],
    'H(m)': [H],
    'Q': [Q],
    'K(MPa)': [K]
})

# Main Panel
st.title("🚇 Tunnel Squeezing Prediction System")
st.markdown("### Machine Learning based Geotechnical Risk Assessment")

st.write("---")

# Create tabs for different views
tab1, tab2, tab3, tab4 = st.tabs([
    "🎯 Single Prediction",
    "🔀 Model Comparison",
    "📊 Batch Processing",
    "ℹ️ Model Information"
])

# Tab 1: Single Prediction
with tab1:
    st.subheader("Single Tunnel Scenario Prediction")
    
    # Model selection
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("#### Model Selection")
        model_choice = st.radio(
            "Choose Model",
            ["Random Forest", "SVM (Support Vector Machine)"],
            help="Select which model to use for prediction"
        )
        
        st.info(f"""
        **Current Input:**
        - **D**: {D} m
        - **H**: {H} m
        - **Q**: {Q}
        - **K**: {K} MPa
        """)
    
    with col2:
        if rf_model is None and svm_model is None:
            st.error("❌ No models loaded. Please check .pkl files.")
        else:
            # Select model
            if model_choice == "Random Forest" and rf_model:
                model = rf_model
            elif model_choice == "SVM (Support Vector Machine)" and svm_model:
                model = svm_model
            else:
                st.error(f"❌ {model_choice} model not available")
                model = None
            
            if model:
                # Get prediction with confidence
                result = get_prediction_with_confidence(model, input_df, model_choice)
                
                if result['success']:
                    # Display prediction with color coding
                    if result['class'] == 1:
                        st.success(f"### 🟢 Prediction: {result['class_name']}")
                    elif result['class'] == 2:
                        st.warning(f"### 🟡 Prediction: {result['class_name']}")
                    else:
                        st.error(f"### 🔴 Prediction: {result['class_name']}")
                    
                    # Display confidence gauge
                    fig_gauge = create_confidence_gauge(result['confidence'], result['class_name'])
                    st.plotly_chart(fig_gauge, use_container_width=True, key="tab1_gauge")
                    
                    # Display probability bars
                    fig_bars = create_probability_bars(result['probabilities'])
                    st.plotly_chart(fig_bars, use_container_width=True, key="tab1_bars")
                    
                    # Display detailed probabilities
                    st.markdown("#### Detailed Probabilities")
                    prob_df = pd.DataFrame({
                        'Class': ['Non-squeezing', 'Minor Squeezing', 'Severe Squeezing'],
                        'Probability': [f"{p*100:.2f}%" for p in result['probabilities']]
                    })
                    st.dataframe(prob_df, use_container_width=True, hide_index=True)
                    
                else:
                    st.error(f"❌ Prediction failed: {result['error']}")
    
    st.write("---")
    
    # Feature visualization
    st.markdown("#### Input Feature Analysis")
    fig_features = create_feature_ranges_plot(D, H, Q, K)
    st.plotly_chart(fig_features, use_container_width=True, key="tab1_features")
    
    # Engineering interpretation
    with st.expander("📖 Engineering Interpretation & Guidelines"):
        st.markdown("""
        ### Squeezing Classes Explained
        
        #### 🟢 Class 1: Non-squeezing
        - **Strain**: < 1%
        - **Behavior**: Stable tunnel conditions
        - **Support**: Standard rock support (rock bolts, shotcrete)
        - **Risk**: Low - minimal convergence expected
        
        #### 🟡 Class 2: Minor Squeezing
        - **Strain**: 1% - 2.5%
        - **Behavior**: Moderate time-dependent deformation
        - **Support**: Enhanced monitoring, possible yielding elements
        - **Risk**: Medium - convergence requires management
        
        #### 🔴 Class 3: Severe Squeezing
        - **Strain**: ≥ 2.5%
        - **Behavior**: Heavy flowing ground, large deformations
        - **Support**: Yielding support, deformable lining, possible forepoling
        - **Risk**: High - significant engineering challenges
        
        ### Input Parameters Guide
        
        **D (Tunnel Diameter)**:
        - Larger diameter = more unstable (larger excavation span)
        - Typical range: 3-15m for civil tunnels
        
        **H (Overburden Depth)**:
        - Greater depth = higher stress = more squeezing potential
        - Critical parameter for squeezing assessment
        
        **Q (Rock Mass Quality)**:
        - Lower Q = weaker rock = higher squeezing risk
        - Barton's Q-index: comprehensive rock mass rating
        
        **K (Rock Mass Stiffness)**:
        - Lower stiffness = more deformable = higher squeezing
        - Represents rock's resistance to deformation
        """)

# Tab 2: Model Comparison
with tab2:
    st.subheader("Model Comparison View")
    st.markdown("Compare predictions from both Random Forest and SVM models side-by-side")
    
    if rf_model and svm_model:
        # Get predictions from both models
        rf_result = get_prediction_with_confidence(rf_model, input_df, "Random Forest")
        svm_result = get_prediction_with_confidence(svm_model, input_df, "SVM")
        
        if rf_result['success'] and svm_result['success']:
            # Create comparison columns
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 🌲 Random Forest")
                if rf_result['class'] == 1:
                    st.success(f"**Prediction:** {rf_result['class_name']}")
                elif rf_result['class'] == 2:
                    st.warning(f"**Prediction:** {rf_result['class_name']}")
                else:
                    st.error(f"**Prediction:** {rf_result['class_name']}")
                
                st.metric("Confidence", f"{rf_result['confidence']:.1f}%")
                
                fig_rf = create_confidence_gauge(rf_result['confidence'], rf_result['class_name'])
                st.plotly_chart(fig_rf, use_container_width=True, key="tab2_rf_gauge")
            
            with col2:
                st.markdown("### 🎯 SVM")
                if svm_result['class'] == 1:
                    st.success(f"**Prediction:** {svm_result['class_name']}")
                elif svm_result['class'] == 2:
                    st.warning(f"**Prediction:** {svm_result['class_name']}")
                else:
                    st.error(f"**Prediction:** {svm_result['class_name']}")
                
                st.metric("Confidence", f"{svm_result['confidence']:.1f}%")
                
                fig_svm = create_confidence_gauge(svm_result['confidence'], svm_result['class_name'])
                st.plotly_chart(fig_svm, use_container_width=True, key="tab2_svm_gauge")
            
            # Agreement/Disagreement Analysis
            st.write("---")
            st.markdown("### 📊 Comparison Analysis")
            
            if rf_result['class'] == svm_result['class']:
                st.success(f"✅ **Models Agree** - Both predict: {rf_result['class_name']}")
                
                # Determine which model is more confident
                if rf_result['confidence'] > svm_result['confidence']:
                    st.info(f"Random Forest is more confident ({rf_result['confidence']:.1f}% vs {svm_result['confidence']:.1f}%)")
                elif svm_result['confidence'] > rf_result['confidence']:
                    st.info(f"SVM is more confident ({svm_result['confidence']:.1f}% vs {rf_result['confidence']:.1f}%)")
                else:
                    st.info("Both models have equal confidence")
            else:
                st.warning(f"⚠️ **Models Disagree** - RF: {rf_result['class_name']} ({rf_result['confidence']:.1f}%) vs SVM: {svm_result['class_name']} ({svm_result['confidence']:.1f}%)")
                st.info("When models disagree, consider:\n- Use the more confident prediction\n- Investigate input parameters more carefully\n- Consult with geotechnical experts")
            
            # Detailed comparison table
            st.markdown("### 📋 Detailed Comparison Table")
            
            comparison_data = {
                'Model': ['Random Forest', 'SVM'],
                'Predicted Class': [rf_result['class_name'], svm_result['class_name']],
                'Confidence': [f"{rf_result['confidence']:.2f}%", f"{svm_result['confidence']:.2f}%"],
                'Non-squeezing': [f"{rf_result['probabilities'][0]*100:.2f}%", f"{svm_result['probabilities'][0]*100:.2f}%"],
                'Minor Squeezing': [f"{rf_result['probabilities'][1]*100:.2f}%", f"{svm_result['probabilities'][1]*100:.2f}%"],
                'Severe Squeezing': [f"{rf_result['probabilities'][2]*100:.2f}%", f"{svm_result['probabilities'][2]*100:.2f}%"]
            }
            
            comparison_df = pd.DataFrame(comparison_data)
            st.dataframe(comparison_df, use_container_width=True, hide_index=True)
            
            # Probability comparison visualization
            st.markdown("### 📊 Probability Distribution Comparison")
            
            fig_comparison = go.Figure()
            
            class_names = ['Non-squeezing', 'Minor Squeezing', 'Severe Squeezing']
            
            fig_comparison.add_trace(go.Bar(
                name='Random Forest',
                x=class_names,
                y=rf_result['probabilities'] * 100,
                marker_color='#1f77b4',
                text=[f"{p*100:.1f}%" for p in rf_result['probabilities']],
                textposition='auto',
            ))
            
            fig_comparison.add_trace(go.Bar(
                name='SVM',
                x=class_names,
                y=svm_result['probabilities'] * 100,
                marker_color='#ff7f0e',
                text=[f"{p*100:.1f}%" for p in svm_result['probabilities']],
                textposition='auto',
            ))
            
            fig_comparison.update_layout(
                barmode='group',
                title="Model Probability Comparison",
                xaxis_title="Squeezing Class",
                yaxis_title="Probability (%)",
                height=400,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            
            st.plotly_chart(fig_comparison, use_container_width=True, key="tab2_comparison")
            
        else:
            if not rf_result['success']:
                st.error(f"Random Forest prediction failed: {rf_result['error']}")
            if not svm_result['success']:
                st.error(f"SVM prediction failed: {svm_result['error']}")
    else:
        if not rf_model:
            st.error("❌ Random Forest model not loaded")
        if not svm_model:
            st.error("❌ SVM model not loaded")

# Tab 3: Batch Processing
with tab3:
    st.subheader("Batch Prediction from CSV")
    st.markdown("Upload a CSV file with multiple tunnel scenarios for batch prediction")
    
    # Show expected format
    with st.expander("📋 Expected CSV Format"):
        st.markdown("""
        Your CSV file should have the following columns (with these exact names):
        - `D (m)` - Tunnel diameter in meters
        - `H(m)` - Overburden depth in meters
        - `Q` - Rock mass quality index
        - `K(MPa)` - Rock mass stiffness in MPa
        
        Example CSV content:
        ```
        D (m),H(m),Q,K(MPa)
        6.0,150,0.4,26.19
        5.8,350,0.5,2.53
        12.0,220,0.8,32.89
        ```
        """)
        
        # Provide sample CSV download
        sample_data = pd.DataFrame({
            'D (m)': [6.0, 5.8, 12.0],
            'H(m)': [150, 350, 220],
            'Q': [0.4, 0.5, 0.8],
            'K(MPa)': [26.19, 2.53, 32.89]
        })
        
        csv_sample = sample_data.to_csv(index=False)
        st.download_button(
            label="Download Sample CSV",
            data=csv_sample,
            file_name="sample_tunnel_data.csv",
            mime="text/csv"
        )
    
    # File uploader
    uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
    
    if uploaded_file is not None:
        try:
            # Read CSV
            batch_df = pd.read_csv(uploaded_file)
            
            st.success(f"✅ File uploaded successfully! {len(batch_df)} scenarios found.")
            
            # Display uploaded data
            st.markdown("#### Uploaded Data Preview")
            st.dataframe(batch_df.head(10), use_container_width=True)
            
            # Validate columns
            required_cols = ['D (m)', 'H(m)', 'Q', 'K(MPa)']
            if not all(col in batch_df.columns for col in required_cols):
                st.error(f"❌ Missing required columns. Expected: {required_cols}")
            else:
                # Model selection for batch
                batch_model_choice = st.selectbox(
                    "Select Model for Batch Prediction",
                    ["Random Forest", "SVM (Support Vector Machine)", "Both Models"]
                )
                
                if st.button("🚀 Run Batch Prediction"):
                    results_list = []
                    
                    with st.spinner("Processing predictions..."):
                        for idx, row in batch_df.iterrows():
                            row_data = pd.DataFrame({
                                'D (m)': [row['D (m)']],
                                'H(m)': [row['H(m)']],
                                'Q': [row['Q']],
                                'K(MPa)': [row['K(MPa)']]
                            })
                            
                            result_row = {
                                'Scenario': idx + 1,
                                'D (m)': row['D (m)'],
                                'H(m)': row['H(m)'],
                                'Q': row['Q'],
                                'K(MPa)': row['K(MPa)']
                            }
                            
                            # Random Forest prediction
                            if batch_model_choice in ["Random Forest", "Both Models"] and rf_model:
                                rf_pred = get_prediction_with_confidence(rf_model, row_data)
                                if rf_pred['success']:
                                    result_row['RF_Prediction'] = rf_pred['class_name']
                                    result_row['RF_Confidence'] = f"{rf_pred['confidence']:.1f}%"
                            
                            # SVM prediction
                            if batch_model_choice in ["SVM (Support Vector Machine)", "Both Models"] and svm_model:
                                svm_pred = get_prediction_with_confidence(svm_model, row_data)
                                if svm_pred['success']:
                                    result_row['SVM_Prediction'] = svm_pred['class_name']
                                    result_row['SVM_Confidence'] = f"{svm_pred['confidence']:.1f}%"
                            
                            results_list.append(result_row)
                        
                        results_df = pd.DataFrame(results_list)
                    
                    st.success("✅ Batch prediction completed!")
                    
                    # Display results
                    st.markdown("#### Prediction Results")
                    st.dataframe(results_df, use_container_width=True, hide_index=True)
                    
                    # Summary statistics
                    st.markdown("#### Summary Statistics")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Total Scenarios", len(results_df))
                    
                    if 'RF_Prediction' in results_df.columns:
                        with col2:
                            rf_class_dist = results_df['RF_Prediction'].value_counts()
                            st.write("**RF Class Distribution:**")
                            for class_name, count in rf_class_dist.items():
                                st.write(f"- {class_name}: {count}")
                    
                    if 'SVM_Prediction' in results_df.columns:
                        with col3:
                            svm_class_dist = results_df['SVM_Prediction'].value_counts()
                            st.write("**SVM Class Distribution:**")
                            for class_name, count in svm_class_dist.items():
                                st.write(f"- {class_name}: {count}")
                    
                    # Download results
                    st.markdown("#### Download Results")
                    csv_results = results_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Results as CSV",
                        data=csv_results,
                        file_name="tunnel_predictions_results.csv",
                        mime="text/csv"
                    )
                    
                    # Visualization of class distribution
                    if 'RF_Prediction' in results_df.columns or 'SVM_Prediction' in results_df.columns:
                        st.markdown("#### Class Distribution Visualization")
                        
                        fig_dist = go.Figure()
                        
                        if 'RF_Prediction' in results_df.columns:
                            rf_counts = results_df['RF_Prediction'].value_counts()
                            fig_dist.add_trace(go.Bar(
                                name='Random Forest',
                                x=rf_counts.index,
                                y=rf_counts.values,
                                marker_color='#1f77b4'
                            ))
                        
                        if 'SVM_Prediction' in results_df.columns:
                            svm_counts = results_df['SVM_Prediction'].value_counts()
                            fig_dist.add_trace(go.Bar(
                                name='SVM',
                                x=svm_counts.index,
                                y=svm_counts.values,
                                marker_color='#ff7f0e'
                            ))
                        
                        fig_dist.update_layout(
                            barmode='group',
                            title="Class Distribution in Batch Predictions",
                            xaxis_title="Squeezing Class",
                            yaxis_title="Count",
                            height=400
                        )
                        
                        st.plotly_chart(fig_dist, use_container_width=True, key="tab3_batch_dist")
        
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")

# Tab 4: Model Information
with tab4:
    st.subheader("Model Information & Performance Metrics")
    
    # Model performance section
    st.markdown("### 📊 Model Performance")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🌲 Random Forest")
        st.markdown("""
        **Algorithm**: Ensemble of Decision Trees
        
        **Key Characteristics:**
        - Combines multiple decision trees
        - Voting mechanism for final prediction
        - Less prone to overfitting
        - Provides feature importance rankings
        
        **Performance Metrics:**
        - **Accuracy**: 87%
        - **Balanced Accuracy**: 85%
        - **F1-Score (macro)**: 0.84
        
        **Strengths:**
        - Good generalization
        - Robust to outliers
        - Interpretable feature importance
        
        **Best For:**
        - General purpose predictions
        - When feature importance is needed
        - Datasets with complex interactions
        """)
    
    with col2:
        st.markdown("#### 🎯 SVM (Support Vector Machine)")
        st.markdown("""
        **Algorithm**: Support Vector Classification with RBF kernel
        
        **Key Characteristics:**
        - Uses RBF (Radial Basis Function) kernel
        - SMOTE for class balancing
        - Optimized hyperparameters (C, gamma)
        - Margin-based classification
        
        **Performance Metrics:**
        - **Accuracy**: 87%
        - **Balanced Accuracy**: 86%
        - **F1-Score (macro)**: 0.85
        
        **Strengths:**
        - Excellent boundary definition
        - Works well with limited data
        - Effective in high-dimensional spaces
        
        **Best For:**
        - Clear class separation
        - When margin maximization is important
        - Small to medium datasets
        """)
    
    st.write("---")
    
    # Feature importance (for Random Forest)
    st.markdown("### 🎯 Feature Importance (Random Forest)")
    
    if rf_model:
        try:
            # Get feature importances
            if hasattr(rf_model, 'feature_importances_'):
                importances = rf_model.feature_importances_
                feature_names = ['D (m)', 'H(m)', 'Q', 'K(MPa)']
                
                # Create bar chart
                fig_importance = go.Figure(go.Bar(
                    x=importances,
                    y=feature_names,
                    orientation='h',
                    marker=dict(
                        color=importances,
                        colorscale='Viridis',
                        showscale=True
                    ),
                    text=[f"{imp:.3f}" for imp in importances],
                    textposition='auto',
                ))
                
                fig_importance.update_layout(
                    title="Feature Importance in Random Forest Model",
                    xaxis_title="Importance Score",
                    yaxis_title="Feature",
                    height=300,
                    margin=dict(l=20, r=20, t=40, b=20)
                )
                
                st.plotly_chart(fig_importance, use_container_width=True, key="tab4_importance")
                
                # Interpretation
                st.info(f"""
                **Most Important Feature**: {feature_names[np.argmax(importances)]} (importance: {np.max(importances):.3f})
                
                This indicates which parameters have the strongest influence on squeezing prediction.
                Higher importance means the feature has more impact on the model's decisions.
                """)
            else:
                st.warning("Feature importances not available for this model.")
        except Exception as e:
            st.error(f"Could not display feature importance: {str(e)}")
    
    st.write("---")
    
    # Model recommendations
    st.markdown("### 💡 When to Use Which Model")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.success("""
        **Use Random Forest when:**
        - You need to understand feature importance
        - You want robust predictions with outliers
        - You prefer interpretable ensemble methods
        - The scenario has complex feature interactions
        """)
    
    with col2:
        st.info("""
        **Use SVM when:**
        - You want maximum accuracy on test data
        - Clear class boundaries are important
        - You're working with well-characterized data
        - Margin-based classification is preferred
        """)
    
    st.write("---")
    
    # Limitations and assumptions
    st.markdown("### ⚠️ Model Limitations & Assumptions")
    
    with st.expander("Click to view important limitations"):
        st.markdown("""
        ### Dataset Limitations
        - **Small dataset**: ~115 tunnel case studies
        - **Class imbalance**: Class 2 (Minor Squeezing) is underrepresented
        - **Geographic bias**: May not represent all geological conditions globally
        
        ### Model Assumptions
        - **Static prediction**: Does not account for time-dependent behavior
        - **Limited features**: Only 4 input parameters (D, H, Q, K)
        - **Simplified classification**: Real squeezing is a continuous spectrum
        - **No site-specific factors**: Local geology, construction methods not included
        
        ### Recommendations for Use
        1. **Use as preliminary screening** - Not a replacement for detailed geotechnical analysis
        2. **Validate with site data** - Compare predictions with local experience
        3. **Consult experts** - Especially for critical or unusual projects
        4. **Consider uncertainty** - Low confidence predictions require more investigation
        5. **Update regularly** - Retrain models as more data becomes available
        
        ### Engineering Judgment
        These predictions should be combined with:
        - Site investigation data
        - Geological mapping
        - In-situ stress measurements
        - Experience from nearby projects
        - Expert geotechnical assessment
        """)
    
    st.write("---")
    
    # Technical details
    st.markdown("### 🔧 Technical Details")
    
    with st.expander("Model Training Information"):
        st.markdown("""
        ### Data Preprocessing
        - **Scaling**: Features standardized/normalized
        - **Class Balancing**: SMOTE (Synthetic Minority Over-sampling Technique) applied
        - **Train-Test Split**: Stratified split to maintain class distribution
        
        ### Hyperparameter Tuning
        - **Method**: Grid Search with Cross-Validation
        - **CV Folds**: 5-fold stratified cross-validation
        - **Scoring Metric**: Balanced accuracy to handle class imbalance
        
        ### Model Persistence
        - **Format**: Joblib (.pkl files)
        - **Version**: Compatible with scikit-learn >= 1.3.0
        - **Size**: Optimized for fast loading
        
        ### Software Stack
        - **Python**: 3.8+
        - **Scikit-learn**: 1.3.0+
        - **Streamlit**: 1.28.0+
        - **Plotly**: 5.17.0+ (for visualizations)
        """)

st.write("---")

# Footer
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**Author:** Sudip Adhikari")

with col2:
    st.markdown("**Powered by:** Streamlit & Scikit-Learn")

with col3:
    st.markdown("**Version:** 2.0 (Enhanced)")
