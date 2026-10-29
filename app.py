import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

# 1. Page Configuration
st.set_page_config(page_title="Sri Lanka Flood Risk Predictor", layout="wide")
st.title("🌊 Sri Lanka District Flood Risk Predictor")
st.write("Predict emergency flood risk levels based on daily ECMWF ERA5 meteorological conditions.")

# 2. Load the Tuned Model and Scaler
@st.cache_resource
def load_assets():
    # Updated to match your exact .joblib filenames
    model = joblib.load('tuned_extra_trees_model.joblib')
    scaler = joblib.load('flood_risk_scaler.joblib')
    return model, scaler

try:
    model, scaler = load_assets()
except Exception as e:
    st.error(f"Error loading files: {e}. Ensure 'tuned_extra_trees_model.joblib' and 'flood_risk_scaler.joblib' are in the directory.")
    st.stop()

# 3. User Input Sidebar
st.sidebar.header("Input Meteorological Data")

# Categorical Inputs
districts = ['Ampara', 'Anuradhapura', 'Badulla', 'Batticaloa', 'Colombo', 'Galle', 'Gampaha', 'Hambantota', 'Jaffna', 'Kalutara', 'Kandy', 'Kegalle', 'Kilinochchi', 'Kurunegala', 'Mannar', 'Matale', 'Matara', 'Monaragala', 'Mullaitivu', 'Nuwara Eliya', 'Polonnaruwa', 'Puttalam', 'Ratnapura', 'Trincomalee', 'Vavuniya']
provinces = ['Central', 'Eastern', 'Northern', 'North Central', 'North Western', 'Sabaragamuwa', 'Southern', 'Uva', 'Western']
climatic_zones = ['Dry', 'Intermediate', 'Wet']

district = st.sidebar.selectbox("District", districts)
province = st.sidebar.selectbox("Province", provinces)
climatic_zone = st.sidebar.selectbox("Climatic Zone", climatic_zones)

st.sidebar.divider()

# Continuous Inputs
precip_sum = st.sidebar.number_input("Precipitation Sum (mm)", min_value=0.0, value=15.0)
rain_48h = st.sidebar.number_input("48-Hour Rain (mm)", min_value=0.0, value=25.0)
temp_max = st.sidebar.number_input("Max Temperature (C)", min_value=10.0, value=30.0)
wind_speed = st.sidebar.number_input("Max Wind Speed (10m)", min_value=0.0, value=15.0)
soil_moisture_0_7 = st.sidebar.number_input("Soil Moisture (0-7cm)", min_value=0.0, max_value=1.0, value=0.3)
soil_moisture_7_28 = st.sidebar.number_input("Soil Moisture (7-28cm)", min_value=0.0, max_value=1.0, value=0.3)
soil_saturation = st.sidebar.number_input("Soil Saturation Index", min_value=0.0, max_value=1.0, value=0.5)

# 4. Process Inputs when User Clicks Predict
if st.button("Predict Flood Risk", type="primary"):
    
    # Construct raw dataframe
    input_data = {
        'district': district, 'province': province, 'climatic_zone': climatic_zone,
        'precipitation_sum': precip_sum, 'soil_moisture_0_to_7cm_mean': soil_moisture_0_7,
        'soil_moisture_7_to_28cm_mean': soil_moisture_7_28, 'temperature_2m_max': temp_max,
        'wind_speed_10m_max': wind_speed, 'rain_48h': rain_48h, 'soil_saturation_index': soil_saturation
    }
    raw_df = pd.DataFrame([input_data])
    
    # Apply One-Hot Encoding strictly matching the training setup
    encoded_df = pd.get_dummies(raw_df, columns=['district', 'province', 'climatic_zone'], dtype=int)
    
    # Align columns to match the exact feature names the model was trained on
    model_features = model.feature_names_in_
    final_df = pd.DataFrame(columns=model_features)
    for col in model_features:
        final_df.loc[0, col] = encoded_df.loc[0, col] if col in encoded_df.columns else 0
        
    final_df = final_df.astype(float)
    
    # Scale continuous variables
    continuous_cols = [
        'precipitation_sum', 'soil_moisture_0_to_7cm_mean', 'soil_moisture_7_to_28cm_mean', 
        'temperature_2m_max', 'wind_speed_10m_max', 'rain_48h', 'soil_saturation_index'
    ]
    final_df[continuous_cols] = scaler.transform(final_df[continuous_cols])
    
    # 5. Execute Prediction
    prediction = model.predict(final_df)[0]
    probabilities = model.predict_proba(final_df)[0]
    
    classes = [
        "Low Risk (Normal)", 
        "Advisory (Waterlogging)", 
        "High Warning (Inundation)", 
        "Critical Emergency (Catastrophic Overflow)"
    ]
    
    # 6. Display Results
    st.subheader(f"🚨 Predicted Status: **{classes[prediction]}**")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Low Risk Prob", f"{probabilities[0]:.1%}")
    col2.metric("Advisory Prob", f"{probabilities[1]:.1%}")
    col3.metric("High Warning Prob", f"{probabilities[2]:.1%}")
    col4.metric("Critical Prob", f"{probabilities[3]:.1%}")
    
    # 7. Dynamic SHAP Explainability
    st.divider()
    st.subheader("Model Explainability (SHAP)")
    st.write("This plot shows exactly how much each weather feature contributed to pushing this specific prediction toward or away from a Critical Emergency.")
    
    with st.spinner("Generating SHAP explanation..."):
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(final_df)
        
        # Plotting the SHAP force plot for Class 3 (Critical Emergency) using matplotlib
        fig = plt.figure(figsize=(10, 4))
        shap.plots.waterfall(shap.Explanation(
            values=shap_values[:, :, 3][0], 
            base_values=explainer.expected_value[3], 
            data=final_df.iloc[0], 
            feature_names=final_df.columns
        ), show=False)
        st.pyplot(fig)
        plt.clf()