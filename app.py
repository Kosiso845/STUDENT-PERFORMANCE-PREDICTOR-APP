import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the trained model
with open("xgb_student_model.pkl", "rb") as file:
    model = pickle.load(file)

# Set page config
st.set_page_config(page_title="Student Performance Predictor", layout="centered")

# Dark mode styling
st.markdown("""
    <style>
        .main {
            background-color: #0E1117;
            color: white;
        }
        .stTextInput>div>div>input, .stNumberInput>div>div>input {
            background-color: #262730;
            color: white;
        }
        .stSelectbox>div>div>div>div {
            background-color: #262730;
            color: white;
        }
    </style>
""", unsafe_allow_html=True)

# Title
st.title("📊 Student Performance Predictor")

# Sidebar
st.sidebar.header("Input Student Information")

def user_input_features():
    age = st.sidebar.slider('Age', 15, 25, 18)
    study_time = st.sidebar.selectbox('Weekly Study Time (hours)', [1, 2, 3, 4])
    past_failures = st.sidebar.selectbox('Past Class Failures', [0, 1, 2, 3])
    absences = st.sidebar.number_input('Total Absences', min_value=0, max_value=100, value=4)
    health = st.sidebar.slider('Health Status (1 = Very Bad, 5 = Very Good)', 1, 5, 3)
    goout = st.sidebar.slider('Going Out Frequency (1 = Low, 5 = High)', 1, 5, 3)
    freetime = st.sidebar.slider('Free Time (1 = Low, 5 = High)', 1, 5, 3)

    data = {
        'age': age,
        'studytime': study_time,
        'failures': past_failures,
        'absences': absences,
        'health': health,
        'goout': goout,
        'freetime': freetime
    }
    features = pd.DataFrame([data])
    return features

input_df = user_input_features()

# Show user input
st.subheader("Student Data Preview")
st.write(input_df)

# Predict
if st.button("Predict Performance"):
    prediction = model.predict(input_df)
    performance = round(float(prediction[0]), 2)
    st.success(f"🎯 Predicted Student Final Grade: **{performance}**")
