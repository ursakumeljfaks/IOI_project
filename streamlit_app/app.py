import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os
import sys

# Add parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from src.model import HousePriceModel
from src.explainer import LIMEExplainer

# Page config
st.set_page_config(
    page_title="House Price LIME Dashboard",
    page_icon="🏠",
    layout="wide"
)

# Title
st.title("🏠 Interactive House Price Explanation Dashboard")
st.markdown("### Understanding what drives house prices using LIME")
st.markdown("**Team:** Urša Kumelj, Timen Bobnar, Matija Krigl")
st.markdown("---")

# Load model
@st.cache_resource
@st.cache_resource
def load_model():
    model = HousePriceModel()
    # Use absolute path
    model_path = os.path.join(parent_dir, 'models', 'house_price_model.pkl')
    model.load(model_path)
    return model

@st.cache_data
def load_data():
    # Use absolute path
    data_path = os.path.join(parent_dir, 'data', 'geeksforgeeks', 'california_housing.csv')
    return pd.read_csv(data_path)

try:
    model = load_model()
    df = load_data()
    
    st.success("✓ Model loaded successfully!")
    
    # Sidebar - Input parameters
    st.sidebar.header("🎛️ House Features")
    st.sidebar.markdown("Adjust the sliders to see how price changes")
    
    # Get feature ranges from data
    features = {}
    feature_descriptions = {
        'MedInc': 'Median Income (in $10k)',
        'HouseAge': 'House Age (years)',
        'AveRooms': 'Average Rooms',
        'AveBedrms': 'Average Bedrooms',
        'Population': 'Population',
        'AveOccup': 'Average Occupancy',
        'Latitude': 'Latitude',
        'Longitude': 'Longitude'
    }
    
    for col in df.columns:
        if col != 'MedHouseVal':
            min_val = float(df[col].min())
            max_val = float(df[col].max())
            mean_val = float(df[col].mean())
            
            label = feature_descriptions.get(col, col)
            
            features[col] = st.sidebar.slider(
                label,
                min_value=min_val,
                max_value=max_val,
                value=mean_val,
                help=f"Range: {min_val:.2f} - {max_val:.2f}"
            )
    
    # Create input dataframe
    input_df = pd.DataFrame([features])
    
    # Make prediction
    prediction = model.predict(input_df)[0]
    
    # Display prediction
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("💰 Predicted Price", f"${prediction * 100000:,.0f}")
    with col2:
        st.metric("📊 Median Dataset Price", f"${df['MedHouseVal'].median() * 100000:,.0f}")
    with col3:
        diff = ((prediction - df['MedHouseVal'].median()) / df['MedHouseVal'].median()) * 100
        st.metric("📈 vs Median", f"{diff:+.1f}%", delta=f"{diff:+.1f}%")
    
    st.markdown("---")
    
    # Prepare for LIME
    X_train = df.drop('MedHouseVal', axis=1)
    X_train_scaled = model.scaler.transform(X_train)
    
    # Create LIME explainer
    explainer = LIMEExplainer(model, X_train_scaled, model.feature_names)
    
    # Get explanation
    input_scaled = model.scaler.transform(input_df)
    explanation = explainer.explain_instance(input_scaled[0], num_features=8)
    
    # Display LIME explanation
    st.header("📊 Why This Price? (LIME Explanation)")
    st.markdown("LIME shows which features are pushing the price **up** (green) or **down** (red)")
    
    top_features = explainer.get_top_features(explanation, n=8)
    
    # Create horizontal bar chart
    fig = go.Figure()
    
    colors = ['#ef4444' if x < 0 else '#22c55e' for x in top_features['impact']]
    
    fig.add_trace(go.Bar(
        y=top_features['feature'],
        x=top_features['impact'],
        orientation='h',
        marker=dict(color=colors),
        text=[f"{x:+.3f}" for x in top_features['impact']],
        textposition='outside',
        hovertemplate='<b>%{y}</b><br>Impact: %{x:.3f}<extra></extra>'
    ))
    
    fig.update_layout(
        title="Feature Impact on Prediction",
        xaxis_title="Impact on Price (in $100k units)",
        yaxis_title="",
        height=400,
        showlegend=False,
        yaxis=dict(autorange="reversed"),
        plot_bgcolor='white'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Explanation text
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **🟢 Green bars (Positive Impact)**
        - Features that INCREASE the predicted price
        - Longer bar = Stronger positive effect
        """)
    with col2:
        st.markdown("""
        **🔴 Red bars (Negative Impact)**
        - Features that DECREASE the predicted price
        - Longer bar = Stronger negative effect
        """)
    
    # Feature values table
    with st.expander("📋 See Current Feature Values"):
        display_df = input_df.T.rename(columns={0: 'Value'})
        display_df['Description'] = display_df.index.map(feature_descriptions)
        st.dataframe(display_df[['Description', 'Value']], use_container_width=True)
    
    # What-if analysis section
    st.markdown("---")
    st.header("🔮 What-If Analysis")
    st.markdown("See how changing specific features affects the price prediction")
    
    selected_feature = st.selectbox(
        "Select a feature to analyze:",
        options=model.feature_names,
        format_func=lambda x: feature_descriptions.get(x, x)
    )
    
    # Generate what-if scenarios
    feature_min = df[selected_feature].quantile(0.05)
    feature_max = df[selected_feature].quantile(0.95)
    feature_values = np.linspace(feature_min, feature_max, 50)
    
    predictions = []
    for val in feature_values:
        temp_input = input_df.copy()
        temp_input[selected_feature] = val
        pred = model.predict(temp_input)[0]
        predictions.append(pred * 100000)
    
    # Plot what-if
    fig_whatif = go.Figure()
    fig_whatif.add_trace(go.Scatter(
        x=feature_values,
        y=predictions,
        mode='lines',
        name='Predicted Price',
        line=dict(color='#3b82f6', width=3)
    ))
    
    # Add current value marker
    current_val = features[selected_feature]
    current_pred = prediction * 100000
    fig_whatif.add_trace(go.Scatter(
        x=[current_val],
        y=[current_pred],
        mode='markers',
        name='Current Selection',
        marker=dict(size=15, color='#ef4444', symbol='star')
    ))
    
    fig_whatif.update_layout(
        title=f"Impact of {feature_descriptions.get(selected_feature, selected_feature)} on Price",
        xaxis_title=feature_descriptions.get(selected_feature, selected_feature),
        yaxis_title="Predicted Price ($)",
        height=400,
        hovermode='x unified',
        plot_bgcolor='white'
    )
    
    st.plotly_chart(fig_whatif, use_container_width=True)

except FileNotFoundError:
    st.error("⚠️ Model not found! Please run the training notebook first.")
    st.info("Run: `jupyter notebook` and execute `notebooks/02_train_model.ipynb`")
except Exception as e:
    st.error(f"❌ Error: {str(e)}")
    import traceback
    st.code(traceback.format_exc())
