"""
Wellness Tourism Package Predictor - Streamlit Application
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from huggingface_hub import hf_hub_download

st.set_page_config(page_title="Wellness Tourism Predictor", page_icon="🌴", layout="wide")

st.title("🌴 Wellness Tourism Package Predictor")
st.markdown("Predict whether a customer will purchase the Wellness Tourism Package")

@st.cache_resource
def load_model():
    HF_USERNAME = "ansee178"
    MODEL_REPO = f"{HF_USERNAME}/wellness-tourism-model"
    
    try:
        model_path = hf_hub_download(repo_id=MODEL_REPO, filename="best_model.joblib")
        encoders_path = hf_hub_download(repo_id=MODEL_REPO, filename="label_encoders.joblib")
        features_path = hf_hub_download(repo_id=MODEL_REPO, filename="feature_names.joblib")
        
        model = joblib.load(model_path)
        encoders = joblib.load(encoders_path)
        features = joblib.load(features_path)
        
        return model, encoders, features, None
    except Exception as e:
        return None, None, None, str(e)

model, encoders, feature_names, error = load_model()

if error:
    st.error(f"Error loading model: {error}")
    st.stop()
else:
    st.success("✅ Model loaded successfully!")

st.divider()
st.header("📝 Enter Customer Details")

col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=35)
    gender = st.selectbox("Gender", ["Male", "Female"])
    city_tier = st.selectbox("City Tier", [1, 2, 3])
    marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced", "Unmarried"])
    occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer"])
    designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])

with col2:
    monthly_income = st.number_input("Monthly Income", min_value=10000, max_value=500000, value=50000)
    num_trips = st.number_input("Annual Trips", min_value=0, max_value=20, value=2)
    num_persons = st.number_input("Persons Visiting", min_value=1, max_value=10, value=2)
    num_children = st.number_input("Children Under 5", min_value=0, max_value=5, value=0)
    passport = st.selectbox("Has Passport?", ["No", "Yes"])
    own_car = st.selectbox("Owns Car?", ["No", "Yes"])
    preferred_star = st.selectbox("Hotel Rating", [3, 4, 5])

with col3:
    type_of_contact = st.selectbox("Contact Type", ["Company Invited", "Self Enquiry"])
    product_pitched = st.selectbox("Product Pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])
    pitch_score = st.slider("Pitch Satisfaction", 1, 5, 3)
    num_followups = st.number_input("Follow-ups", min_value=1, max_value=10, value=3)
    duration_pitch = st.number_input("Pitch Duration (min)", min_value=5, max_value=60, value=15)

st.divider()

if st.button("🔮 Predict Purchase Likelihood", type="primary", use_container_width=True):
    input_data = {
        "Age": age, "TypeofContact": type_of_contact, "CityTier": city_tier,
        "DurationOfPitch": duration_pitch, "Occupation": occupation, "Gender": gender,
        "NumberOfPersonVisiting": num_persons, "NumberOfFollowups": num_followups,
        "ProductPitched": product_pitched, "PreferredPropertyStar": preferred_star,
        "MaritalStatus": marital_status, "NumberOfTrips": num_trips,
        "Passport": 1 if passport == "Yes" else 0, "PitchSatisfactionScore": pitch_score,
        "OwnCar": 1 if own_car == "Yes" else 0, "NumberOfChildrenVisiting": num_children,
        "Designation": designation, "MonthlyIncome": monthly_income
    }
    
    input_df = pd.DataFrame([input_data])
    
    for col, encoder in encoders.items():
        if col in input_df.columns:
            try:
                input_df[col] = encoder.transform(input_df[col].astype(str))
            except:
                input_df[col] = 0
    
    input_df = input_df.reindex(columns=feature_names, fill_value=0)
    
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0]
    
    st.header("🎯 Prediction Results")
    col_a, col_b = st.columns(2)
    
    with col_a:
        if prediction == 1:
            st.success("### ✅ LIKELY TO PURCHASE")
        else:
            st.warning("### ⚠️ UNLIKELY TO PURCHASE")
    
    with col_b:
        st.metric("Purchase Probability", f"{probability[1]*100:.1f}%")
        st.progress(probability[1])

st.divider()
st.markdown("**Built with Streamlit** | MLOps Pipeline Project")
