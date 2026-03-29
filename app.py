import streamlit as st
import numpy as np
import joblib

# Load model and scaler
model = joblib.load("svm_heart_model.pkl")
scaler = joblib.load("svm_scaler.pkl")

st.title("❤️ Heart Attack Prediction (SVM Model)")

st.write("Enter Patient Details")

# Input fields
age = st.number_input("Age")
sex = st.selectbox("Sex", [0,1])
cp = st.number_input("Chest Pain Type (0-3)")
trestbps = st.number_input("Resting Blood Pressure")
chol = st.number_input("Cholesterol")
fbs = st.selectbox("Fasting Blood Sugar >120", [0,1])
restecg = st.number_input("Rest ECG (0-2)")
thalach = st.number_input("Max Heart Rate")
exang = st.selectbox("Exercise Induced Angina", [0,1])
oldpeak = st.number_input("Oldpeak")
slope = st.number_input("Slope (0-2)")
ca = st.number_input("Number of Vessels (0-3)")
thal = st.number_input("Thal (0-3)")

# Prediction button
if st.button("Predict"):

    input_data = np.array([[age,sex,cp,trestbps,chol,fbs,
                            restecg,thalach,exang,oldpeak,
                            slope,ca,thal]])

    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)

    st.subheader("Result")

    if prediction[0] == 1:
        st.error("⚠ High Risk of Heart Attack")
    else:
        st.success("✅ Low Risk of Heart Attack")