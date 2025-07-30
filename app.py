import streamlit as st
import pandas as pd
import pickle

# Force dark styling via CSS (in addition to config.toml)
st.markdown("""
    <style>
        html, body, [class*="css"]  {
            background-color: #0e1117 !important;
            color: #fafafa !important;
        }
        .stButton>button {
            background-color: #00c4ff;
            color: white;
        }
    </style>
""", unsafe_allow_html=True)

# Load trained model
with open("xgb_student_model.pkl", "rb") as file:
    model = pickle.load(file)

# All features used in the model
expected_features = [
    'age', 'Medu', 'Fedu', 'traveltime', 'studytime', 'failures', 'famrel',
    'freetime', 'goout', 'Dalc', 'Walc', 'health', 'absences',
    'sex_M', 'address_U', 'famsize_LE3', 'Pstatus_T', 'Mjob_health',
    'Mjob_other', 'Mjob_services', 'Mjob_teacher', 'Fjob_health', 'Fjob_other',
    'Fjob_services', 'Fjob_teacher', 'reason_home', 'reason_other',
    'reason_reputation', 'schoolsup_yes', 'famsup_yes', 'paid_yes',
    'activities_yes', 'nursery_yes', 'higher_yes', 'internet_yes', 'romantic_yes'
]

# Streamlit app title
st.title("🎓 Student Performance Prediction")

# User inputs
st.subheader("Enter Student Information")

age = st.slider("Age", 15, 22)
studytime = st.slider("Weekly Study Time (1–4)", 1, 4)
failures = st.number_input("Past Class Failures", min_value=0, max_value=5, step=1)
absences = st.number_input("Total Absences", min_value=0, step=1)
health = st.slider("Health (1 = Very Bad, 5 = Very Good)", 1, 5)
goout = st.slider("Going Out Frequency (1 = Low, 5 = High)", 1, 5)
freetime = st.slider("Free Time (1 = Low, 5 = High)", 1, 5)
Dalc = st.slider("Workday Alcohol Consumption (1–5)", 1, 5)
Walc = st.slider("Weekend Alcohol Consumption (1–5)", 1, 5)
Medu = st.slider("Mother's Education (0–4)", 0, 4)
Fedu = st.slider("Father's Education (0–4)", 0, 4)

# Create base input
input_data = {
    'age': age,
    'studytime': studytime,
    'failures': failures,
    'absences': absences,
    'health': health,
    'goout': goout,
    'freetime': freetime,
    'Dalc': Dalc,
    'Walc': Walc,
    'Medu': Medu,
    'Fedu': Fedu
}

# Convert to DataFrame
input_df = pd.DataFrame([input_data])

# Fill in missing features with 0
for feature in expected_features:
    if feature not in input_df.columns:
        input_df[feature] = 0

# Reorder columns to match model input
input_df = input_df[expected_features]

# Predict
if st.button("Predict Performance"):
    prediction = model.predict(input_df)[0]
    st.success(f"📊 Predicted Outcome: **{prediction}**")
