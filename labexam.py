# Step 4: Build Streamlit App
import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("heart_disease_model.pkl")

st.title("Heart Disease Risk Prediction")

# Collect user input
BMI = st.slider("BMI", 10.0, 50.0, 25.0)
Smoking = st.selectbox("Do you smoke?", ["Yes", "No"])
AlcoholDrinking = st.selectbox("Do you drink alcohol?", ["Yes", "No"])
Stroke = st.selectbox("History of stroke?", ["Yes", "No"])
PhysicalHealth = st.slider("Days of poor physical health (0–30)", 0, 30, 0)
MentalHealth = st.slider("Days of poor mental health (0–30)", 0, 30, 0)
SleepTime = st.slider("Average hours of sleep", 1, 24, 7)
DiffWalking = st.selectbox("Difficulty walking?", ["Yes", "No"])
Sex = st.selectbox("Sex", ["Male", "Female"])
AgeCategory = st.selectbox("Age Category", [
    "18-24", "25-29", "30-34", "35-39", "40-44", "45-49",
    "50-54", "55-59", "60-64", "65-69", "70-74", "75-79", "80 or older"])
Race = st.selectbox("Race", ["White", "Black", "Asian", "American Indian/Alaskan Native", "Other", "Hispanic"])
Diabetic = st.selectbox("Diabetic?", ["Yes", "No", "No, borderline diabetes", "Yes (during pregnancy)"])
PhysicalActivity = st.selectbox("Physically Active?", ["Yes", "No"])
GenHealth = st.selectbox("General Health", ["Poor", "Fair", "Good", "Very good", "Excellent"])
Asthma = st.selectbox("Asthma?", ["Yes", "No"])
KidneyDisease = st.selectbox("Kidney Disease?", ["Yes", "No"])
SkinCancer = st.selectbox("Skin Cancer?", ["Yes", "No"])

# When user clicks "Predict"
if st.button("Predict"):

    # Prepare input dictionary (you must match this with training feature names!)
    input_dict = {
        'BMI': BMI,
        'PhysicalHealth': PhysicalHealth,
        'MentalHealth': MentalHealth,
        'SleepTime': SleepTime,
        'Smoking_Yes': 1 if Smoking == "Yes" else 0,
        'AlcoholDrinking_Yes': 1 if AlcoholDrinking == "Yes" else 0,
        'Stroke_Yes': 1 if Stroke == "Yes" else 0,
        'DiffWalking_Yes': 1 if DiffWalking == "Yes" else 0,
        'Sex_Male': 1 if Sex == "Male" else 0,
        'AgeCategory_80 or older': 1 if AgeCategory == "80 or older" else 0,  # add more if needed
        'Race_White': 1 if Race == "White" else 0,  # repeat for each category used in training
        'Diabetic_Yes': 1 if Diabetic == "Yes" else 0,
        'PhysicalActivity_Yes': 1 if PhysicalActivity == "Yes" else 0,
        'GenHealth_Excellent': 1 if GenHealth == "Excellent" else 0,  # add other levels
        'Asthma_Yes': 1 if Asthma == "Yes" else 0,
        'KidneyDisease_Yes': 1 if KidneyDisease == "Yes" else 0,
        'SkinCancer_Yes': 1 if SkinCancer == "Yes" else 0
    }

    # Convert to DataFrame (ensure correct column order as in training!)
    input_df = pd.DataFrame([input_dict])

    # Match model features by reindexing with default 0
    model_features = model.feature_names_in_
    input_df = input_df.reindex(columns=model_features, fill_value=0)

    # Make prediction
    prediction = model.predict(input_df)[0]
    confidence = model.predict_proba(input_df)[0][prediction]

    # Show result
    st.success("Prediction: At Risk" if prediction == 1 else "Not At Risk")
    st.info(f"Confidence Score: {confidence:.2f}")
