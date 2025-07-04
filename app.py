import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import plotly.express as px
import os

# Cache model and scaler to avoid reloading each time
@st.cache_resource
def load_ann_model():
    return load_model("models/retail_sales_ann_model.h5")

@st.cache_resource
def load_scaler():
    return joblib.load("models/scaler.pkl")

@st.cache_resource
def load_feature_columns():
    with open("models/feature_columns.pkl", "rb") as f:
        return joblib.load(f)

# Load the model and artifacts
model = load_ann_model()
scaler = load_scaler()
feature_columns = load_feature_columns()

st.title("🛒 Retail Sales Forecasting with Markdown Impact")

# File uploader
uploaded_file = st.file_uploader("📁 Upload input CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # Check for required columns
    if not set(feature_columns).issubset(df.columns):
        st.error("❌ Uploaded file doesn't match required features.")
    else:
        X = df[feature_columns]
        X_scaled = scaler.transform(X)
        predictions = model.predict(X_scaled).flatten()
        df['Predicted_Weekly_Sales'] = predictions

        # Show sample predictions
        st.subheader("📊 Sample Predictions")
        st.write(df[['Store', 'Dept', 'Date', 'Predicted_Weekly_Sales']].head())

        # Show evaluation metrics if actuals are present
        if 'Weekly_Sales' in df.columns:
            y_true = df['Weekly_Sales']
            y_pred = df['Predicted_Weekly_Sales']

            st.subheader("📈 Evaluation Metrics")
            st.write(f"**MAE**: {mean_absolute_error(y_true, y_pred):.2f}")
            st.write(f"**MSE**: {mean_squared_error(y_true, y_pred):.2f}")
            st.write(f"**RMSE**: {np.sqrt(mean_squared_error(y_true, y_pred)):.2f}")
            st.write(f"**R² Score**: {r2_score(y_true, y_pred):.4f}")

        # Visualize markdown impact
        st.subheader("📉 Markdown Impact on Sales")
        if 'Total_MarkDown' not in df.columns:
            markdown_cols = ['MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5']
            df['Total_MarkDown'] = df[markdown_cols].sum(axis=1)

        # Rename column if needed
        if 'IsHoliday_y' in df.columns:
            df.rename(columns={'IsHoliday_y': 'IsHoliday'}, inplace=True)

        # Plot
        fig = px.scatter(
            df,
            x='Total_MarkDown',
            y='Predicted_Weekly_Sales',
            color='IsHoliday' if 'IsHoliday' in df.columns else None,
            title='Markdown vs Predicted Sales',
            trendline='ols'
        )
        st.plotly_chart(fig, use_container_width=True)

        # Save results
        os.makedirs("data/predictions", exist_ok=True)
        output_path = "data/predictions/streamlit_predictions.csv"
        df.to_csv(output_path, index=False)
        st.success(f"✅ Predictions saved to: `{output_path}`")
