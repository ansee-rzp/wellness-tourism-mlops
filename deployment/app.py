import streamlit as st
import pandas as pd
import joblib
from huggingface_hub import hf_hub_download

st.set_page_config(page_title="Wellness Tourism Predictor", page_icon="🌴", layout="wide")
st.title("🌴 Wellness Tourism Package Predictor")
st.markdown("Predict whether a customer will purchase the Wellness Tourism Package")

@st.cache_resource
def load_model():
    MODEL_REPO = "ansee178/wellness-tourism-model"
    model = joblib.load(hf_hub_download(repo_id=MODEL_REPO, filename="best_model.joblib"))
    features = joblib.load(hf_hub_download(repo_id=MODEL_REPO, filename="feature_names.joblib"))
    return model, features

try:
    model, feature_names = load_model()
    st.success("✅ Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

st.divider()
st.header("📝 Enter Customer Details")

col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("Age", 18, 100, 35)
    city_tier = st.selectbox("City Tier", [1, 2, 3])
    occupation = st.selectbox("Occupation", [0, 1, 2, 3], 
        format_func=lambda x: ["Free Lancer", "Large Business", "Salaried", "Small Business"][x])
    gender = st.selectbox("Gender", [0, 1], format_func=lambda x: ["Female", "Male"][x])
    marital = st.selectbox("Marital Status", [0, 1, 2, 3], 
        format_func=lambda x: ["Divorced", "Married", "Single", "Unmarried"][x])
    designation = st.selectbox("Designation", [0, 1, 2, 3, 4], 
        format_func=lambda x: ["AVP", "Executive", "Manager", "Senior Manager", "VP"][x])

with col2:
    income = st.number_input("Monthly Income (₹)", 10000, 500000, 50000, step=5000)
    trips = st.number_input("Annual Trips", 0, 20, 2)
    persons = st.number_input("Persons Visiting", 1, 10, 2)
    children = st.number_input("Children (Under 5)", 0, 5, 0)
    passport = st.selectbox("Has Passport?", [0, 1], format_func=lambda x: ["No", "Yes"][x])
    own_car = st.selectbox("Owns Car?", [0, 1], format_func=lambda x: ["No", "Yes"][x])

with col3:
    contact = st.selectbox("Type of Contact", [0, 1], 
        format_func=lambda x: ["Company Invited", "Self Enquiry"][x])
    product = st.selectbox("Product Pitched", [0, 1, 2, 3, 4], 
        format_func=lambda x: ["Basic", "Deluxe", "King", "Standard", "Super Deluxe"][x])
    star = st.selectbox("Preferred Hotel Star", [3.0, 4.0, 5.0])
    pitch_score = st.slider("Pitch Satisfaction Score", 1, 5, 3)
    followups = st.number_input("Number of Follow-ups", 1, 10, 3)
    duration = st.number_input("Pitch Duration (min)", 5, 60, 15)

st.divider()

if st.button("🔮 Predict Purchase Likelihood", type="primary", use_container_width=True):
    # Create input in same order as training features
    input_dict = {
        "Age": age,
        "TypeofContact": contact,
        "CityTier": city_tier,
        "DurationOfPitch": float(duration),
        "Occupation": occupation,
        "Gender": gender,
        "NumberOfPersonVisiting": persons,
        "NumberOfFollowups": float(followups),
        "ProductPitched": product,
        "PreferredPropertyStar": star,
        "MaritalStatus": marital,
        "NumberOfTrips": float(trips),
        "Passport": passport,
        "PitchSatisfactionScore": pitch_score,
        "OwnCar": own_car,
        "NumberOfChildrenVisiting": float(children),
        "Designation": designation,
        "MonthlyIncome": float(income)
    }
    
    input_df = pd.DataFrame([input_dict])
    input_df = input_df[feature_names]  # Ensure correct column order
    
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0]
    
    st.header("🎯 Prediction Result")
    col_a, col_b = st.columns(2)
    
    with col_a:
        if prediction == 1:
            st.success("### ✅ LIKELY TO PURCHASE")
            st.markdown("This customer has high potential!")
        else:
            st.warning("### ⚠️ UNLIKELY TO PURCHASE")
            st.markdown("This customer may need more nurturing.")
    
    with col_b:
        st.metric("Purchase Probability", f"{probability[1]*100:.1f}%")
        st.progress(probability[1])

st.divider()
st.markdown("Built with ❤️ using Streamlit | MLOps Project")
