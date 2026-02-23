import streamlit as st
import pandas as pd
import joblib
import os
from src.data_prep import preprocess_data
import matplotlib.pyplot as plt
import seaborn as sns

# Set page config
st.set_page_config(page_title="Vehicle Maintenance Predictor", layout="wide")

st.title("🚗 Vehicle Maintenance Prediction System")
st.markdown("""
Predict vehicle maintenance needs using Machine Learning. Upload your fleet data to identify high-risk vehicles.
""")

# Sidebar for model info
st.sidebar.header("About")
st.sidebar.info("This system uses a Random Forest model trained on vehicle mileage, age, and service history.")

# File upload
uploaded_file = st.file_uploader("Upload Vehicle Data (CSV)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("Uploaded Data Preview")
    st.write(df.head())
    
    # Load model
    if os.path.exists('models/vehicle_model.pkl'):
        model = joblib.load('models/vehicle_model.pkl')
        
        # Preprocess
        X = preprocess_data(df, is_training=False)
        
        # Ensure we don't include the target if it exists in the uploaded file
        if 'Need_Maintenance' in X.columns:
            X_input = X.drop(columns=['Need_Maintenance'])
        else:
            X_input = X
            
        # Prediction
        predictions = model.predict(X_input)
        probabilities = model.predict_proba(X_input)[:, 1]
        
        # Results
        df['Maintenance_Risk'] = predictions
        df['Risk_Score'] = probabilities
        
        st.subheader("Maintenance Predictions")
        
        # Style result
        def highlight_risk(val):
            color = 'red' if val == 1 else 'green'
            return f'color: {color}'
            
        st.write(df[['Vehicle_Model', 'Mileage', 'Vehicle_Age', 'Maintenance_Risk', 'Risk_Score']].head(20))
        
        # Summary Visuals
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("### Risk Distribution")
            fig, ax = plt.subplots()
            sns.countplot(x='Maintenance_Risk', data=df, ax=ax, palette='viridis')
            ax.set_xticklabels(['Low Risk', 'High Risk'])
            st.pyplot(fig)
            
        with col2:
            st.write("### Factors Importance")
            importances = model.feature_importances_
            feat_importance = pd.DataFrame({'Feature': X_input.columns, 'Importance': importances})
            feat_importance = feat_importance.sort_values('Importance', ascending=False)
            
            fig, ax = plt.subplots()
            sns.barplot(x='Importance', y='Feature', data=feat_importance, ax=ax, palette='magma')
            st.pyplot(fig)
            
        # Download results
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Predictions CSV", csv, "vehicle_predictions.csv", "text/csv")
    else:
        st.error("Model file not found. Please train the model first.")
else:
    st.info("Waiting for CSV file upload...")
    
    # Show example format
    st.write("### Expected CSV Format")
    example_df = pd.DataFrame({
        'Vehicle_Model': ['Sedan', 'SUV'],
        'Mileage': [50000, 120000],
        'Maintenance_History': ['Good', 'Poor'],
        'Reported_Issues': [1, 5],
        'Vehicle_Age': [3, 8],
        'Fuel_Type': ['Petrol', 'Diesel'],
        'Odometer_Reading': [51000, 122000]
    })
    st.table(example_df)
