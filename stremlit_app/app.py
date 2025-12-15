import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os
import sys
import json
import altair as alt

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from src.model import HousePriceModel
from src.explainer import LIMEExplainer

st.set_page_config(
    page_title="House Price App",
    page_icon="",
    layout="wide"
)

page = st.radio("Navigate to:", ["Price Prediction", "Data Exploration"], horizontal=True)

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
    st.error("Model or data not found! Make sure `house_price_model.pkl` and CSV exist.")
    st.stop()
except Exception as e:
    st.error(f"Error loading resources: {str(e)}")
    st.stop()

if page == "Price Prediction":
    st.title("House Price Prediction")
    st.markdown(
        "This section allows you to explore how different house and location features "
        "affect the predicted median house price. Adjust the feature values in the sidebar "
        "and observe how the prediction changes."
    )

    st.sidebar.header("House Features")
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

    feature_info = {
    'MedInc': {
        'label': 'Median Income',
        'help': 'Median income of households in the area, measured in tens of thousands of USD.'
    },
    'HouseAge': {
        'label': 'House Age',
        'help': 'Median age of houses in the area, in years.'
    },
    'AveRooms': {
        'label': 'Average Rooms',
        'help': 'Average number of rooms per household.'
    },
    'AveBedrms': {
        'label': 'Average Bedrooms',
        'help': 'Average number of bedrooms per household.'
    },
    'Population': {
        'label': 'Population',
        'help': 'Number of people living in the area.'
    },
    'AveOccup': {
        'label': 'Average Occupancy',
        'help': 'Average number of people per household.'
    },
    'Latitude': {
        'label': 'Latitude',
        'help': 'Geographic latitude of the location.'
    },
    'Longitude': {
        'label': 'Longitude',
        'help': 'Geographic longitude of the location.'
    }
}
    features = {}

    for col in df.columns:
        if col != 'MedHouseVal':
            min_val = float(df[col].min())
            max_val = float(df[col].max())
            mean_val = float(df[col].mean())

            info = feature_info.get(col, {'label': col, 'help': ''})

            features[col] = st.sidebar.slider(
                label=info['label'],
                min_value=min_val,
                max_value=max_val,
                value=mean_val,
                help=info['help']
            )


    input_df = pd.DataFrame([features])
    prediction = model.predict(input_df)[0]

    col1, col2, col3 = st.columns(3)
    
    #col1.metric("Predicted Price", f"${prediction*100000:,.0f}")
    #col2.metric("Median Dataset Price", f"${df['MedHouseVal'].median()*100000:,.0f}")
    #diff = ((prediction - df['MedHouseVal'].median()) / df['MedHouseVal'].median())*100
    #col3.metric(" vs Median", f"{diff:+.1f}%", delta=f"{diff:+.1f}%")
    
    diff = ((prediction - df['MedHouseVal'].median()) / df['MedHouseVal'].median())*100
    col1.metric(
    "Predicted House Price",
    f"${prediction*100000:,.0f}"
    )

    col2.metric(
        "Median Price Across Dataset",
        f"${df['MedHouseVal'].median()*100000:,.0f}"
    )

    col3.metric(
        "Difference Compared to Dataset Median",
        f"{diff:+.1f}%",
        delta=f"{diff:+.1f}%"
    )

    st.markdown("---")

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
        title="Contribution of Individual Features to the Predicted Price",
        xaxis_title="Estimated Impact on Price (in $100,000 units)",
        yaxis=dict(autorange="reversed"),
        height=400,
        showlegend=False
    )
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Selected Feature Values"):
        display_df = input_df.T.rename(columns={0: 'Value'})
        display_df['Description'] = display_df.index.map(feature_descriptions)
        st.dataframe(display_df[['Description', 'Value']], use_container_width=True)

    st.markdown("---")
    st.header("What-if Feature Analysis")
    selected_feature = st.selectbox(
        "Select a feature to analyze its effect on the predicted price:", options=model.feature_names,
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
        title=(
        f"Effect of {feature_descriptions.get(selected_feature, selected_feature)} on the Predicted House Price"),
        xaxis_title=feature_descriptions.get(selected_feature, selected_feature),
        yaxis_title="Predicted Price ($)",
        height=400,
        hovermode='x unified'
    )
    st.plotly_chart(fig_whatif, use_container_width=True)

elif page == "Data Exploration":
    st.title("Data Exploration with Vega-Lite")

    with st.expander("Distribution of House Prices"):
        st.markdown("This scatter plot illustrates the relationship between median income and median house value. Point color represents population size, providing additional context about density.")
        price_hist = alt.Chart(df).mark_bar().encode(
            alt.X('MedHouseVal', bin=alt.Bin(maxbins=50), title='Median House Value ($100k)'),
            y='count()',
            tooltip=['count()']
        ).properties(height=300, width='container')
        st.altair_chart(price_hist, use_container_width=True)

    with st.expander("Price vs Median Income"):
        st.markdown("This scatter plot shows how median house prices relate to median household income. Each point represents one district. Color intensity reflects population size.")
        scatter_income = alt.Chart(df).mark_circle(size=60).encode(
            x=alt.X(
                'MedInc:Q',
                title='Median Household Income (× $10,000)'
            ),
            y=alt.Y(
                'MedHouseVal:Q',
                title='Median House Price (× $100,000)'
            ),
            color=alt.Color(
                'Population:Q',
                title='Population',
                scale=alt.Scale(scheme='viridis')
            ),
            tooltip=[
                alt.Tooltip('MedInc:Q', title='Median Income (× $10,000)'),
                alt.Tooltip('HouseAge:Q', title='House Age (years)'),
                alt.Tooltip('MedHouseVal:Q', title='House Price (× $100,000)'),
                alt.Tooltip('Population:Q', title='Population')
            ]
        ).interactive().properties(height=300, width='container')
        st.altair_chart(scatter_income, use_container_width=True)

    with st.expander("Geospatial Distribution of Houses in California"):
        st.markdown("This map visualizes the geographic distribution of median house prices across California. Each point represents a district from the dataset, where color indicates the median house value and point size corresponds to population. The state border is shown for geographic reference.")
        geojson_path = os.path.join(current_dir, '..', 'data', 'json', 'california.geojson')
        with open(geojson_path) as f:
            ca_geo_single = json.load(f)

        ca_geo = {"type": "FeatureCollection", "features": [ca_geo_single]}

        ca_layer = (
            alt.Chart(alt.Data(values=[ca_geo['features'][0]]))
            .mark_geoshape(fill=None, stroke="black", strokeWidth=2)
            .project(type="mercator")
        )

        scatter_layer = (
            alt.Chart(df)
            .mark_circle()
            .encode(
                longitude=alt.Longitude(
                    'Longitude:Q',
                    title='Longitude'
                ),
                latitude=alt.Latitude(
                    'Latitude:Q',
                    title='Latitude'
                ),
                color=alt.Color(
                    'MedHouseVal:Q',
                    title='Median House Price (× $100,000)',
                    scale=alt.Scale(scheme='viridis')
                ),
                size=alt.Size(
                    'Population:Q',
                    title='Population',
                    scale=alt.Scale(range=[10, 500])
                ),
                tooltip=[
                    alt.Tooltip('MedHouseVal:Q', title='House Price (× $100,000)'),
                    alt.Tooltip('MedInc:Q', title='Median Income (× $10,000)'),
                    alt.Tooltip('HouseAge:Q', title='House Age (years)'),
                    alt.Tooltip('Population:Q', title='Population')
                ],
                    )
                .project(type="mercator")  
            )

        geo_chart = (
            alt.layer(ca_layer, scatter_layer)
            .properties(width="container", height=600, title="California House Prices with State Border")
            .interactive()
        )

        st.altair_chart(geo_chart, use_container_width=True)


    with st.expander("Feature Correlation Heatmap"):
        st.markdown("This heatmap shows pairwise correlations between numerical features in the dataset. Stronger colors indicate stronger positive or negative relationships, which can help identify dependencies between variables.")
        corr = df.corr().reset_index().melt('index')
        heatmap = alt.Chart(corr).mark_rect().encode(
            x=alt.X('index:O', title='Feature'),
            y=alt.Y('variable:O', title='Feature'),
            color=alt.Color(
                'value:Q',
                title='Correlation Coefficient',
                scale=alt.Scale(scheme='redblue', domainMid=0)
            ),
            tooltip=[
                alt.Tooltip('index:O', title='Feature 1'),
                alt.Tooltip('variable:O', title='Feature 2'),
                alt.Tooltip('value:Q', title='Correlation')
            ]
        )
        st.altair_chart(heatmap, use_container_width=True)

