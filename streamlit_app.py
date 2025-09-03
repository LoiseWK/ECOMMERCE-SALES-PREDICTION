import streamlit as st
import pandas as pd
import pickle
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np

# ====================== 
# Page Configuration
# ======================
st.set_page_config(
    page_title="E-commerce Sales Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)
# Custom CSS for better styling
st.markdown("""
<style>
    .stApp {
        background-color: #0e1117;
        color: #ffffff;
    }
    
    .stAlert > div {
        background-color: #1e2329 !important;
        border: 1px solid #404040 !important;
        color: #ffffff !important;
    }
    
    .stSidebar .stAlert > div {
        background-color: #262626 !important;
        border: 1px solid #404040 !important;
        color: #ffffff !important;
    }
    
    .stMarkdown, .stText, p, div {
        color: #ffffff !important;
    }
    
    h1, h2, h3, h4, h5, h6 {
        color: #ffffff !important;
    }
    
    .stSidebar {
        background-color: #262626;
    }
    
    .stMetric {
        background-color: #1e2329;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #404040;
    }
</style>
""", unsafe_allow_html=True)


# ====================== 
# 1. Load Model & Data 
# ======================
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("Ecommerce_Sales_Prediction_Dataset.csv")
        df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
        df = df.dropna(subset=['Date'])
        return df
    except FileNotFoundError:
        st.error("❌ Dataset file not found. Please ensure 'Ecommerce_Sales_Prediction_Dataset.csv' is in the correct directory.")
        st.stop()

@st.cache_resource
def load_model():
    try:
        import joblib
        return joblib.load("best_model.pkl")
    except FileNotFoundError:
        st.error("❌ Model file not found. Please ensure 'best_model.pkl' is in the correct directory.")
        st.stop()

# Load data and model
with st.spinner("🔄 Loading data and model..."):
    data = load_data()
    model_pipeline = load_model()

# ====================== 
# 2. Header Section
# ======================
st.markdown("""
<div style="
    font-size: 4rem; 
    font-weight: bold;
    color: #4fc3f7; 
    text-align: center; 
    margin: 2rem 0; 
    padding: 2rem;
    background: linear-gradient(135deg, #1a1a2e, #16213e, #0f3460);
    border-radius: 15px;
    border: 2px solid #4fc3f7;
    box-shadow: 0 8px 32px rgba(79, 195, 247, 0.3);
    text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
">
📈 E-COMMERCE SALES FORECASTING DASHBOARD
</div>
""", unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("📦 Total Records", f"{len(data):,}")
with col2:
    st.metric("🏷️ Categories", len(data['Product_Category'].unique()))
with col3:
    st.metric("📅 Date Range", f"{(data['Date'].max() - data['Date'].min()).days} days")
with col4:
    st.metric("💰 Avg Daily Sales", f"{data['Units_Sold'].mean():.0f}")

# ====================== 
# 3. Sidebar Filters 
# ======================
st.sidebar.markdown("## 🔎 Dashboard Filters")

# Add data info in sidebar
st.sidebar.markdown("### 📊 Dataset Info")
st.sidebar.markdown(f"""
<div style="background-color: #262626; padding: 1rem; border-radius: 8px; border: 1px solid #404040; color: #ffffff;">
<strong>Date Range:</strong> {data['Date'].min().strftime('%Y-%m-%d')} to {data['Date'].max().strftime('%Y-%m-%d')}<br>
<strong>Total Sales:</strong> {data['Units_Sold'].sum():,} units<br>
<strong>Peak Sales:</strong> {data['Units_Sold'].max():,} units
</div>
""", unsafe_allow_html=True)

# Month filter
data['Month'] = data['Date'].dt.strftime("%B") 
data['Year'] = data['Date'].dt.year

month_filter = st.sidebar.multiselect(
    "📅 Select Month(s):", 
    options=sorted(data['Month'].unique()),
    default=sorted(data['Month'].unique())
)

# Category filter
category_filter = st.sidebar.multiselect(
    "🏷️ Select Product Category:", 
    options=sorted(data['Product_Category'].unique()),
    default=sorted(data['Product_Category'].unique())
)

# Year filter
year_filter = st.sidebar.multiselect(
    "📆 Select Year(s):",
    options=sorted(data['Year'].unique()),
    default=sorted(data['Year'].unique())
)

# Apply filters
filtered_data = data[
    (data['Month'].isin(month_filter)) & 
    (data['Product_Category'].isin(category_filter)) &
    (data['Year'].isin(year_filter))
]

# ====================== 
# 4. Data Overview 
# ======================
st.markdown("## 📋 Filtered Data Overview")

if len(filtered_data) == 0:
    st.warning("⚠️ No data matches the current filters. Please adjust your selection.")
    st.stop()

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### Sample Data")
    st.dataframe(
        filtered_data.head(10).style.format({'Units_Sold': '{:,.0f}'}),
        use_container_width=True
    )

with col2:
    st.markdown("### Quick Stats")
    st.markdown(f"""
    <div class="metric-container">
        <strong>📊 Filtered Records:</strong> {len(filtered_data):,}<br>
        <strong>💰 Total Units Sold:</strong> {filtered_data['Units_Sold'].sum():,}<br>
        <strong>📈 Average Sales:</strong> {filtered_data['Units_Sold'].mean():.0f}<br>
        <strong>🔝 Peak Sales:</strong> {filtered_data['Units_Sold'].max():,}<br>
        <strong>📉 Min Sales:</strong> {filtered_data['Units_Sold'].min():,}
    </div>
    """, unsafe_allow_html=True)

# ====================== 
# 5. Visualization Section 
# ======================
st.markdown("## 📊 Sales Analytics")

# Create tabs for different views
tab1, tab2, tab3, tab4 = st.tabs(["📈 Time Series", "📊 Category Analysis", "🗓️ Seasonal Patterns", "🎯 Forecast"])

with tab1:
    st.markdown("### Sales Trend Over Time")
    
    # Interactive Plotly chart
    fig = px.line(
        filtered_data, 
        x='Date', 
        y='Units_Sold', 
        color='Product_Category',
        title="Sales Trend by Product Category",
        hover_data=['Month']
    )
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Units Sold",
        hovermode='x unified',
        height=500
    )
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Sales by Category")
        category_sales = filtered_data.groupby('Product_Category')['Units_Sold'].sum().reset_index()
        
        fig = px.bar(
            category_sales, 
            x='Product_Category', 
            y='Units_Sold',
            title="Total Sales by Category",
            color='Units_Sold',
            color_continuous_scale='viridis'
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### Category Distribution")
        fig = px.pie(
            category_sales, 
            values='Units_Sold', 
            names='Product_Category',
            title="Sales Distribution by Category"
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.markdown("### Seasonal Sales Patterns")
    
    # Monthly aggregation
    monthly_data = filtered_data.groupby(['Month', 'Product_Category'])['Units_Sold'].sum().reset_index()
    
    # Heatmap
    pivot_data = monthly_data.pivot(index='Product_Category', columns='Month', values='Units_Sold').fillna(0)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(pivot_data, annot=True, fmt='.0f', cmap='YlOrRd', ax=ax)
    ax.set_title('Sales Heatmap: Category vs Month')
    st.pyplot(fig)

with tab4:
    st.markdown('<div class="forecast-highlight">', unsafe_allow_html=True)
    st.markdown("## 🔮 Sales Forecasting")
    st.markdown("Generate predictions for future sales based on historical data.")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Forecast parameters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        forecast_category = st.selectbox(
            "🏷️ Select Product Category", 
            options=sorted(data['Product_Category'].unique()),
            help="Choose the product category for forecasting"
        )
    
    with col2:
        forecast_horizon = st.slider(
            "📅 Forecast Horizon (days)", 
            min_value=1, 
            max_value=90, 
            value=14,
            help="Number of days to forecast into the future"
        )
    
    with col3:
        confidence_level = st.selectbox(
            "📊 Confidence Level",
            options=[80, 90, 95],
            index=1,
            help="Statistical confidence level for predictions"
        )
    
    # Generate forecast button
    if st.button("🚀 Generate Forecast", type="primary", use_container_width=True):
        with st.spinner(f"🔄 Generating {forecast_horizon}-day forecast for {forecast_category}..."):
            try:
                # Generate future dates
                last_date = data['Date'].max()
                future_dates = pd.date_range(
                    start=last_date + pd.Timedelta(days=1), 
                    periods=forecast_horizon
                )
                
                # Build input DataFrame for prediction
                # Get recent data for the selected category to base predictions on
                recent_data = filtered_data[filtered_data['Product_Category'] == forecast_category].tail(30)

                if len(recent_data) == 0:
                         st.error("No historical data found for this category")
                else:
                # Simple forecasting using historical average (since model needs complex features)
                         avg_sales = recent_data['Units_Sold'].mean()
                         std_sales = recent_data['Units_Sold'].std() if len(recent_data) > 1 else avg_sales * 0.1
    
                # Generate predictions with some trend and seasonality
                np.random.seed(42)
                trend = np.linspace(0, len(recent_data) * 0.01, forecast_horizon)  # Slight upward trend
                seasonal = np.sin(np.arange(forecast_horizon) * 2 * np.pi / 7) * (std_sales * 0.2)  # Weekly pattern
                noise = np.random.normal(0, std_sales * 0.1, forecast_horizon)
    
                y_pred = avg_sales + trend + seasonal + noise
                y_pred = np.abs(y_pred)  # Ensure positive values
                future_df = pd.DataFrame({
                    "Date": future_dates,
                    "Product_Category": forecast_category
                })
                
                # Predict
                y_pred = model_pipeline.predict(future_df)
                future_df["Predicted_Units_Sold"] = y_pred
                
                # Add confidence intervals (simplified simulation)
                np.random.seed(42)
                std_dev = filtered_data[filtered_data['Product_Category'] == forecast_category]['Units_Sold'].std()
                error_margin = std_dev * (confidence_level/100) * 0.2
                
                future_df["Lower_Bound"] = y_pred - error_margin
                future_df["Upper_Bound"] = y_pred + error_margin
                future_df["Lower_Bound"] = future_df["Lower_Bound"].clip(lower=0)  # Ensure non-negative
                
                # Display results
                st.success(f"✅ Forecast generated successfully for {forecast_category}!")
                
                # Forecast summary metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("🎯 Predicted Avg", f"{y_pred.mean():.0f}")
                with col2:
                    st.metric("📈 Predicted Max", f"{y_pred.max():.0f}")
                with col3:
                    st.metric("📉 Predicted Min", f"{y_pred.min():.0f}")
                with col4:
                    st.metric("📊 Total Predicted", f"{y_pred.sum():.0f}")
                
                # Detailed forecast table
                st.markdown("### 📋 Detailed Forecast")
                forecast_display = future_df.copy()
                forecast_display['Date'] = forecast_display['Date'].dt.strftime('%Y-%m-%d')
                forecast_display = forecast_display.round(0)
                
                st.dataframe(
                    forecast_display.style.format({
                        'Predicted_Units_Sold': '{:.0f}',
                        'Lower_Bound': '{:.0f}',
                        'Upper_Bound': '{:.0f}'
                    }),
                    use_container_width=True
                )
                
                # Interactive forecast visualization
                st.markdown("### 📈 Forecast Visualization")
                
                # Get historical data for context
                historical = filtered_data[
                    filtered_data['Product_Category'] == forecast_category
                ].sort_values('Date').tail(30)  # Last 30 days for context
                
                # Create interactive plot
                fig = go.Figure()
                
                # Historical data
                fig.add_trace(go.Scatter(
                    x=historical['Date'],
                    y=historical['Units_Sold'],
                    mode='lines+markers',
                    name='Historical Sales',
                    line=dict(color='#1f77b4', width=2),
                    marker=dict(size=4)
                ))
                
                # Forecast data
                fig.add_trace(go.Scatter(
                    x=future_df['Date'],
                    y=future_df['Predicted_Units_Sold'],
                    mode='lines+markers',
                    name='Predicted Sales',
                    line=dict(color='#ff7f0e', width=2),
                    marker=dict(size=6, symbol='diamond')
                ))
                
                # Confidence intervals
                fig.add_trace(go.Scatter(
                    x=list(future_df['Date']) + list(future_df['Date'][::-1]),
                    y=list(future_df['Upper_Bound']) + list(future_df['Lower_Bound'][::-1]),
                    fill='toself',
                    fillcolor='rgba(255,127,14,0.2)',
                    line=dict(color='rgba(255,127,14,0)'),
                    name=f'{confidence_level}% Confidence Interval',
                    showlegend=True
                ))
                
                fig.update_layout(
                    title=f"Sales Forecast: {forecast_category} ({forecast_horizon} days ahead)",
                    xaxis_title="Date",
                    yaxis_title="Units Sold",
                    hovermode='x unified',
                    height=500,
                    showlegend=True
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Forecast insights
                st.markdown("### Forecast Insights")
                
                avg_historical = historical['Units_Sold'].mean() if len(historical) > 0 else 0
                avg_predicted = y_pred.mean()
                trend = "📈 increasing" if avg_predicted > avg_historical else "📉 decreasing"
                change_pct = ((avg_predicted - avg_historical) / avg_historical * 100) if avg_historical > 0 else 0
                
                insights = f"""
                **Key Insights for {forecast_category}:**
                - The forecast shows a **{trend}** trend compared to recent historical data
                - Predicted average daily sales: **{avg_predicted:.0f} units**
                - Change from historical average: **{change_pct:+.1f}%**
                - Total predicted sales over {forecast_horizon} days: **{y_pred.sum():.0f} units**
                - Confidence level: **{confidence_level}%**
                """
                
                st.info(insights)
                
            except Exception as e:
                st.error(f"❌ Error generating forecast: {str(e)}")
                st.error("Please check your model file and data format.")

# ====================== 
# Footer
# ======================
st.markdown("---")
st.markdown("### 📌 Dashboard Information")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown('<div style="background-color: #1e2329; padding: 1rem; border-radius: 8px; border: 1px solid #404040;"><strong style="color: #ffffff;">🔧 Model:</strong> <span style="color: #ffffff;">Machine Learning Pipeline</span></div>', unsafe_allow_html=True)
with col2:
    st.markdown('<div style="background-color: #1e2329; padding: 1rem; border-radius: 8px; border: 1px solid #404040;"><strong style="color: #ffffff;">📊 Data Source:</strong> <span style="color: #ffffff;">E-commerce Sales Dataset</span></div>', unsafe_allow_html=True)
with col3:
    st.markdown(f'<div style="background-color: #1e2329; padding: 1rem; border-radius: 8px; border: 1px solid #404040;"><strong style="color: #ffffff;">⏰ Last Updated:</strong> <span style="color: #ffffff;">{datetime.now().strftime("%Y-%m-%d %H:%M")}</span></div>', unsafe_allow_html=True)

# Sidebar footer
st.sidebar.markdown("---")
st.sidebar.markdown("### 🎛️ Dashboard Controls")
if st.sidebar.button("🔄 Refresh Data", help="Reload the dashboard data"):
    st.cache_data.clear()
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.markdown("**💡 Tips:**")
st.sidebar.markdown("- Use filters to focus on specific time periods or categories")
st.sidebar.markdown("- Check different tabs for various analytics views")  
st.sidebar.markdown("- Generate forecasts in the Forecast tab")
st.sidebar.markdown("- Hover over charts for detailed information")