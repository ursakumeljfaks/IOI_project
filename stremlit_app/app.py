import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os
import sys
import json
import altair as alt

# Add parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from src.model import HousePriceModel
from src.explainer import LIMEExplainer

# Page config
st.set_page_config(
    page_title="House Price App",
    page_icon="🏠",
    layout="wide"
)

# --- Navigation ---
page = st.radio("Navigate to:", ["🏠 Price Prediction", "📊 Data Exploration"], horizontal=True)

# --- Load resources ---
@st.cache_resource
def load_model():
    model = HousePriceModel()
    model_path = os.path.join(parent_dir, 'models', 'house_price_model.pkl')
    model.load(model_path)
    return model

@st.cache_data
def load_data():
    data_path = os.path.join(parent_dir, 'data', 'geeksforgeeks', 'california_housing.csv')
    return pd.read_csv(data_path)

try:
    model = load_model()
    df = load_data()
except FileNotFoundError:
    st.error("⚠️ Model or data not found! Make sure `house_price_model.pkl` and CSV exist.")
    st.stop()
except Exception as e:
    st.error(f"❌ Error loading resources: {str(e)}")
    st.stop()

# --- Price Prediction Page ---
if page == "🏠 Price Prediction":
    st.title("🏠 Interactive House Price Prediction")
    st.markdown("Adjust house features in the sidebar to see the predicted price.")

    # Sidebar features
    st.sidebar.header("🎛️ House Features")
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

    features = {}
    for col in df.columns:
        if col != 'MedHouseVal':
            min_val, max_val = float(df[col].min()), float(df[col].max())
            mean_val = float(df[col].mean())
            label = feature_descriptions.get(col, col)
            features[col] = st.sidebar.slider(
                label, min_value=min_val, max_value=max_val, value=mean_val,
                help=f"Range: {min_val:.2f} - {max_val:.2f}"
            )

    input_df = pd.DataFrame([features])
    prediction = model.predict(input_df)[0]

    col1, col2, col3 = st.columns(3)
    col1.metric("💰 Predicted Price", f"${prediction*100000:,.0f}")
    col2.metric("📊 Median Dataset Price", f"${df['MedHouseVal'].median()*100000:,.0f}")
    diff = ((prediction - df['MedHouseVal'].median()) / df['MedHouseVal'].median())*100
    col3.metric("📈 vs Median", f"{diff:+.1f}%", delta=f"{diff:+.1f}%")

    st.markdown("---")

    # LIME explanation
    X_train = df.drop('MedHouseVal', axis=1)
    X_train_scaled = model.scaler.transform(X_train)
    explainer = LIMEExplainer(model, X_train_scaled, model.feature_names)
    input_scaled = model.scaler.transform(input_df)
    explanation = explainer.explain_instance(input_scaled[0], num_features=8)
    top_features = explainer.get_top_features(explanation, n=8)

    colors = ['#ef4444' if x < 0 else '#22c55e' for x in top_features['impact']]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=top_features['feature'],
        x=top_features['impact'],
        orientation='h',
        marker=dict(color=colors),
        text=[f"{x:+.3f}" for x in top_features['impact']],
        textposition='outside'
    ))
    fig.update_layout(
        title="Feature Impact on Prediction",
        xaxis_title="Impact on Price (in $100k units)",
        yaxis=dict(autorange="reversed"),
        height=400,
        showlegend=False
    )
    st.plotly_chart(fig, use_container_width=True)

    # Feature table
    with st.expander("📋 Feature Values"):
        display_df = input_df.T.rename(columns={0: 'Value'})
        display_df['Description'] = display_df.index.map(feature_descriptions)
        st.dataframe(display_df[['Description', 'Value']], use_container_width=True)

    # What-if analysis
    st.markdown("---")
    st.header("🔮 What-If Analysis")
    selected_feature = st.selectbox(
        "Feature to analyze:", options=model.feature_names,
        format_func=lambda x: feature_descriptions.get(x, x)
    )
    feature_min = df[selected_feature].quantile(0.05)
    feature_max = df[selected_feature].quantile(0.95)
    feature_values = np.linspace(feature_min, feature_max, 50)
    predictions = [model.predict(input_df.assign(**{selected_feature: val}))[0]*100000 for val in feature_values]

    fig_whatif = go.Figure()
    fig_whatif.add_trace(go.Scatter(x=feature_values, y=predictions, mode='lines', line=dict(color='#3b82f6', width=3)))
    fig_whatif.add_trace(go.Scatter(x=[features[selected_feature]], y=[prediction*100000], mode='markers', marker=dict(size=15, color='#ef4444', symbol='star')))
    fig_whatif.update_layout(
        title=f"Impact of {feature_descriptions.get(selected_feature, selected_feature)} on Price",
        xaxis_title=feature_descriptions.get(selected_feature, selected_feature),
        yaxis_title="Predicted Price ($)",
        height=400,
        hovermode='x unified'
    )
    st.plotly_chart(fig_whatif, use_container_width=True)

# --- Data Exploration Page ---
elif page == "📊 Data Exploration":
    st.title("📊 Data Exploration with Vega-Lite")

    # 1. Price distribution
    with st.expander("🏠 Distribution of House Prices"):
        st.markdown("This histogram shows how median house prices are distributed across California. You can see which price ranges are most common.")
        price_hist = alt.Chart(df).mark_bar().encode(
            alt.X('MedHouseVal', bin=alt.Bin(maxbins=50), title='Median House Value ($100k)'),
            y='count()',
            tooltip=['count()']
        ).properties(height=300, width='container')
        st.altair_chart(price_hist, use_container_width=True)

    # 2. Price vs Median Income
    with st.expander("💰 Price vs Median Income"):
        st.markdown("This scatter plot shows the relationship between median income and house prices. Color indicates population, showing areas with more people.")
        scatter_income = alt.Chart(df).mark_circle(size=60).encode(
            x='MedInc',
            y='MedHouseVal',
            color='Population',
            tooltip=['MedInc', 'HouseAge', 'MedHouseVal']
        ).interactive().properties(height=300, width='container')
        st.altair_chart(scatter_income, use_container_width=True)

    # 3. Geospatial (California)
    with st.expander("🗺️ Geospatial Distribution of Houses in California"):
        st.markdown("This map shows house prices across California...")

        # Load California geojson
        geojson_path = os.path.join(current_dir, '..', 'data', 'json', 'california.geojson')
        with open(geojson_path) as f:
            ca_geo_single = json.load(f)

        ca_geo = {"type": "FeatureCollection", "features": [ca_geo_single]}

        # Base CA shape
        ca_layer = (
            alt.Chart(alt.Data(values=[ca_geo['features'][0]]))
            .mark_geoshape(fill=None, stroke="black", strokeWidth=2)
            .project(type="mercator")
        )

        # Scatter layer using same projection
        scatter_layer = (
            alt.Chart(df)
            .mark_circle()
            .encode(
                longitude='Longitude:Q',
                latitude='Latitude:Q',
                color=alt.Color('MedHouseVal:Q', scale=alt.Scale(scheme='viridis')),
                size=alt.Size('Population:Q', scale=alt.Scale(range=[10, 500])),
                tooltip=['MedHouseVal', 'MedInc', 'HouseAge', 'Population'],
            )
            .project(type="mercator")   # <-- KEY FIX
        )

        geo_chart = (
            alt.layer(ca_layer, scatter_layer)
            .properties(width="container", height=600, title="California House Prices with State Border")
            .interactive()
        )

        st.altair_chart(geo_chart, use_container_width=True)


    # 4. Correlation heatmap
    with st.expander("📈 Feature Correlation Heatmap"):
        st.markdown("This heatmap shows how features relate to each other. Darker colors indicate stronger positive or negative correlations, helping identify patterns in the dataset.")
        corr = df.corr().reset_index().melt('index')
        heatmap = alt.Chart(corr).mark_rect().encode(
            x='index:O',
            y='variable:O',
            color='value:Q',
            tooltip=['index', 'variable', 'value']
        ).properties(height=300, width='container')
        st.altair_chart(heatmap, use_container_width=True)
