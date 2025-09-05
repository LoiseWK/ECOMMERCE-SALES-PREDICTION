import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import seaborn as sns


# Page setup with dark theme
st.set_page_config(page_title="Sales Dashboard", layout="wide")

# Dark theme CSS
st.markdown("""
<style>
    .stApp {
        background-color: #1e1e1e;
        color: #ffffff;
    }
    
    .stMarkdown, .stText {
        color: #ffffff !important;
    }
    
    .stMetric {
        background-color: #2d2d2d;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #404040;
    }
    
    .stMetric .metric-container {
        background-color: #2d2d2d;
    }
    
    .stDataFrame {
        background-color: #2d2d2d;
    }
    
    .stSelectbox, .stSlider, .stButton {
        background-color: #2d2d2d;
        color: #ffffff;
    }
    
    h1, h2, h3, h4, h5, h6 {
        color: #ffffff !important;
    }
    
    .stSidebar {
        background-color: #262626;
    }
    
    .stSidebar .stMarkdown {
        color: #ffffff !important;
    }
    
    /* Fix info boxes */
    .stAlert > div {
        background-color: #2d2d2d !important;
        border: 1px solid #404040 !important;
        color: #ffffff !important;
    }
</style>
""", unsafe_allow_html=True)

# Load data
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("Ecommerce_Sales_Prediction_Dataset.csv")
        df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
        return df.dropna(subset=['Date'])
    except FileNotFoundError:
        st.error("❌ Dataset file not found. Please ensure 'Ecommerce_Sales_Prediction_Dataset.csv' is in the correct directory.")
        st.stop()

@st.cache_resource
def load_model():
    try:
        return joblib.load("best_model.pkl")
    except FileNotFoundError:
        st.error("❌ Model file not found. Please ensure 'best_model.pkl' is in the correct directory.")
        st.stop()

# Feature engineering function
def create_features(df, category, forecast_horizon, last_date):
    """Create feature dataframe with all required features for the model"""
    
    # Generate future dates
    future_dates = pd.date_range(
        start=last_date + pd.Timedelta(days=1), 
        periods=forecast_horizon
    )
    
    # Get recent data for the category
    recent_data = df[df['Product_Category'] == category].sort_values('Date').tail(30)
    
    if len(recent_data) == 0:
        return None
    
    # Create base dataframe
    feature_df = pd.DataFrame({
        'Date': future_dates,
        'Product_Category': category
    })
    
    # Get last known values and averages
    last_values = recent_data.iloc[-1]
    avg_values = recent_data.mean(numeric_only=True)
    
    # Price and marketing features (use defaults if not available)
    feature_df['Price'] = last_values.get('Price', 100)
    feature_df['Discount'] = last_values.get('Discount', 0.1)
    feature_df['Marketing_Spend'] = last_values.get('Marketing_Spend', 1000)
    
    
    most_common_segment = recent_data['Customer_Segment'].mode()
    feature_df['Customer_Segment'] = most_common_segment.iloc[0] if len(most_common_segment) > 0 else 'Regular'
    
    # Create lag features
    feature_df['lag_1'] = recent_data['Units_Sold'].iloc[-1]
    feature_df['lag_7'] = recent_data['Units_Sold'].iloc[-7] if len(recent_data) >= 7 else recent_data['Units_Sold'].iloc[-1]
    feature_df['lag_14'] = recent_data['Units_Sold'].iloc[-14] if len(recent_data) >= 14 else recent_data['Units_Sold'].iloc[-1]
    
    # Rolling statistics
    feature_df['roll_mean_7'] = recent_data['Units_Sold'].tail(7).mean()
    feature_df['roll_std_7'] = recent_data['Units_Sold'].tail(7).std()
    feature_df['roll_mean_14'] = recent_data['Units_Sold'].tail(14).mean() if len(recent_data) >= 14 else feature_df['roll_mean_7'].iloc[0]
    feature_df['roll_std_14'] = recent_data['Units_Sold'].tail(14).std() if len(recent_data) >= 14 else feature_df['roll_std_7'].iloc[0]
    feature_df['roll_mean_30'] = recent_data['Units_Sold'].mean()
    feature_df['roll_std_30'] = recent_data['Units_Sold'].std()
    
    # Exponential weighted moving averages
    ewm_7 = recent_data['Units_Sold'].ewm(span=7).mean().iloc[-1]
    ewm_14 = recent_data['Units_Sold'].ewm(span=14).mean().iloc[-1]
    ewm_30 = recent_data['Units_Sold'].ewm(span=30).mean().iloc[-1]
    
    feature_df['ewm_mean_7'] = ewm_7
    feature_df['ewm_mean_14'] = ewm_14
    feature_df['ewm_mean_30'] = ewm_30
    
    # Date features
    feature_df['dayofweek'] = feature_df['Date'].dt.dayofweek
    feature_df['month'] = feature_df['Date'].dt.month
    feature_df['Year'] = feature_df['Date'].dt.year
    feature_df['WeekOfYear'] = feature_df['Date'].dt.isocalendar().week
    feature_df['Quarter'] = feature_df['Date'].dt.quarter
    feature_df['DayOfYear'] = feature_df['Date'].dt.dayofyear
    feature_df['is_weekend'] = (feature_df['Date'].dt.dayofweek >= 5).astype(int)
    feature_df['IsMonthEnd'] = feature_df['Date'].dt.is_month_end.astype(int)
    feature_df['IsMonthStart'] = feature_df['Date'].dt.is_month_start.astype(int)
    
    # Holiday features (simplified)
    feature_df['is_holiday'] = 0
    feature_df['days_to_holiday'] = 30
    
    # Fill any remaining NaN values
    feature_df = feature_df.fillna(method='ffill').fillna(0)
    
    return feature_df

# Initialize
data = load_data()
model = load_model()

# Header with bigger title
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

# Sidebar filters
st.sidebar.header("Filters")

data['Month'] = data['Date'].dt.strftime("%B")
data['Year'] = data['Date'].dt.year
months = st.sidebar.multiselect("Months:", sorted(data['Month'].unique()), sorted(data['Month'].unique()))
categories = st.sidebar.multiselect("Categories:", sorted(data['Product_Category'].unique()), sorted(data['Product_Category'].unique()))
years = st.sidebar.multiselect("Years:", sorted(data['Year'].unique()), sorted(data['Year'].unique()))
# Sidebar info
st.sidebar.markdown("### 📊 Dataset Info")
st.sidebar.markdown(f""" 
<div style="background-color: #262626; padding: 1rem; border-radius: 8px; border: 1px solid #404040; color: #ffffff;">
<strong>Date Range:</strong> {data['Date'].min().strftime('%Y-%m-%d')} to {data['Date'].max().strftime('%Y-%m-%d')}<br>
<strong>Total Sales:</strong> {data['Units_Sold'].sum():,} units<br>
<strong>Peak Sales:</strong> {data['Units_Sold'].max():,} units
</div>
""", unsafe_allow_html=True)

# Filter data
filtered_data = data[
    (data['Month'].isin(months)) & 
    (data['Product_Category'].isin(categories)) & 
    (data['Year'].isin(years))
]

# Check if filtered data is empty
if len(filtered_data) == 0:
    st.warning("⚠️ No data matches the current filters. Please adjust your selection.")
    st.stop()

# Main dashboard metrics
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total Records", f"{len(filtered_data):,}")
with col2:
    st.metric("Total Sales", f"{filtered_data['Units_Sold'].sum():,}")
with col3:
    st.metric("Avg Daily Sales", f"{filtered_data['Units_Sold'].mean():.0f}")

# Data preview
st.subheader("Data Preview")
st.dataframe(filtered_data.head(10))

# Visualization Section


# Create tabs for different views
tab1, tab2, tab3, tab4 = st.tabs(["📈 Time Series", "📊 Category Analysis", "🗓️ Seasonal Patterns", "🎯 Forecast"])

st.markdown("## 📊 Sales Analytics")

# Time Series Analysis
with tab1:
# Sales trend chart
     
     st.subheader("Sales Trend")
     fig, ax = plt.subplots(figsize=(10, 6), facecolor='#1e1e1e')
     ax.set_facecolor('#2d2d2d')

     for cat in filtered_data['Product_Category'].unique():
        subset = filtered_data[filtered_data['Product_Category'] == cat]
        ax.plot(subset['Date'], subset['Units_Sold'], label=cat, marker='o', linewidth=2)

     ax.set_xlabel("Date", color='white')
     ax.set_ylabel("Units Sold", color='white')
     ax.tick_params(colors='white')
     ax.legend(facecolor='#2d2d2d', edgecolor='white', labelcolor='white')
     ax.grid(True, alpha=0.3, color='white')
     ax.spines['bottom'].set_color('white')
     ax.spines['top'].set_color('white')
     ax.spines['right'].set_color('white')
     ax.spines['left'].set_color('white')
     st.pyplot(fig)

# Category Analysis
with tab2:
    st.subheader("Category Analysis")
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

# Seasonal Patterns
with tab3:
    monthly_data = filtered_data.groupby(['Month', 'Product_Category'])['Units_Sold'].sum().reset_index()
    
    # Heatmap
    pivot_data = monthly_data.pivot(index='Product_Category', columns='Month', values='Units_Sold').fillna(0)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(pivot_data, annot=True, fmt='.0f', cmap='YlOrRd', ax=ax)
    ax.set_title('Sales Heatmap: Category vs Month')
    st.pyplot(fig)

# Forecasting section
with tab4:
    st.subheader("🔮 Sales Forecast")

    col1, col2 = st.columns(2)
    with col1:
       forecast_category = st.selectbox("Category:", sorted(data['Product_Category'].unique()))
    with col2:
       forecast_days = st.slider("Forecast Days:", 1, 30, 7)

    if st.button("Generate Forecast", type="primary"):
       with st.spinner(f"Generating {forecast_days}-day forecast..."):
        try:
            # Get last date
            last_date = data['Date'].max()
            
            # Create feature dataframe
            future_df = create_features(data, forecast_category, forecast_days, last_date)
            
            if future_df is None:
                st.error("❌ No historical data found for this category.")
            else:
                # Try model prediction first
                try:
                    predictions = model.predict(future_df)
                    st.success("✅ Model predictions generated!")
                    prediction_method = "Machine Learning Model"
                    
                except Exception as model_error:
                    st.warning(f"⚠️ Model failed: {str(model_error)[:100]}...")
                    st.info("🔄 Using statistical forecasting instead...")
                    
                    # Fallback to statistical method
                    recent_data = filtered_data[filtered_data['Product_Category'] == forecast_category].tail(20)
                    if len(recent_data) > 0:
                        avg_sales = recent_data['Units_Sold'].mean()
                        std_sales = recent_data['Units_Sold'].std()
                        
                        np.random.seed(42)
                        trend = np.linspace(0, avg_sales * 0.02, forecast_days)
                        seasonal = np.sin(np.arange(forecast_days) * 2 * np.pi / 7) * (std_sales * 0.1)
                        noise = np.random.normal(0, std_sales * 0.05, forecast_days)
                        
                        predictions = avg_sales + trend + seasonal + noise
                        predictions = np.abs(predictions)
                        prediction_method = "Statistical Forecasting"
                    else:
                        st.error("No historical data available for statistical forecasting.")
                        st.stop()
                
                # Add predictions to dataframe
                future_df["Predicted_Sales"] = predictions
                
                # Show results
                st.success(f"✅ Forecast generated using: **{prediction_method}**")
                
                # Forecast metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Avg Predicted", f"{predictions.mean():.0f}")
                with col2:
                    st.metric("Max Predicted", f"{predictions.max():.0f}")
                with col3:
                    st.metric("Total Predicted", f"{predictions.sum():.0f}")
                
                # Forecast table
                st.subheader("Forecast Details")
                display_df = future_df[['Date', 'Predicted_Sales']].copy()
                display_df['Date'] = display_df['Date'].dt.strftime('%Y-%m-%d')
                display_df['Predicted_Sales'] = display_df['Predicted_Sales'].round(0)
                st.dataframe(display_df)
                
                # Forecast visualization
                st.subheader("Forecast Chart")
                
                # Get recent historical data for context
                historical = data[data['Product_Category'] == forecast_category].tail(20)
                
                fig, ax = plt.subplots(figsize=(12, 6), facecolor='#1e1e1e')
                ax.set_facecolor('#2d2d2d')
                
                # Plot historical
                if len(historical) > 0:
                    ax.plot(historical['Date'], historical['Units_Sold'], 
                            'o-', label='Historical', color='#00d4aa', linewidth=2, markersize=6)
                
                # Plot forecast
                ax.plot(future_df['Date'], future_df['Predicted_Sales'], 
                        'o-', label='Forecast', color='#ff6b6b', linewidth=2, markersize=6)
                
                ax.axvline(x=last_date, color='#ffd93d', linestyle='--', alpha=0.8, linewidth=2, label='Forecast Start')
                ax.set_xlabel("Date", color='white', fontsize=12)
                ax.set_ylabel("Units Sold", color='white', fontsize=12)
                ax.set_title(f"Sales Forecast: {forecast_category} ({prediction_method})", color='white', fontsize=14)
                ax.tick_params(colors='white')
                ax.legend(facecolor='#2d2d2d', edgecolor='white', labelcolor='white')
                ax.grid(True, alpha=0.3, color='white')
                
                # Set spine colors
                for spine in ax.spines.values():
                    spine.set_color('white')
                
                plt.xticks(rotation=45, color='white')
                plt.tight_layout()
                st.pyplot(fig)
                
                # Forecast insights
                if len(historical) > 0:
                    historical_avg = historical['Units_Sold'].mean()
                    predicted_avg = predictions.mean()
                    change_pct = ((predicted_avg - historical_avg) / historical_avg * 100) if historical_avg > 0 else 0
                    trend_direction = "📈 increasing" if change_pct > 0 else "📉 decreasing"
                    
                    st.subheader("💡 Forecast Insights")
                    st.info(f"""
                    **Key Insights for {forecast_category}:**
                    - Forecast shows a **{trend_direction}** trend ({change_pct:+.1f}% change)
                    - Average predicted daily sales: **{predicted_avg:.0f} units**
                    - Total predicted sales: **{predictions.sum():.0f} units**
                    - Prediction method: **{prediction_method}**
                    """)
        
        except Exception as e:
            st.error(f"❌ Error generating forecast: {str(e)}")
            st.error("Please check your data and try again.")


# Footer
st.markdown("---")
st.markdown("### 📌 Dashboard Information")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown('<div style="background-color: #1e2329; padding: 1rem; border-radius: 8px; border: 1px solid #404040;"><strong style="color: #ffffff;">🔧 Model:</strong> <span style="color: #ffffff;">Machine Learning Pipeline</span></div>', unsafe_allow_html=True)
with col2:
    st.markdown('<div style="background-color: #1e2329; padding: 1rem; border-radius: 8px; border: 1px solid #404040;"><strong style="color: #ffffff;">📊 Data Source:</strong> <span style="color: #ffffff;">E-commerce Sales Dataset</span></div>', unsafe_allow_html=True)
with col3:
    st.markdown(f'<div style="background-color: #1e2329; padding: 1rem; border-radius: 8px; border: 1px solid #404040;"><strong style="color: #ffffff;">⏰ Last Updated:</strong> <span style="color: #ffffff;">Just Now</span></div>', unsafe_allow_html=True)