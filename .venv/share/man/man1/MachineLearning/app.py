import streamlit as st
import pandas as pd
import joblib
import os


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "wine_quality_best_model.pkl")

model = joblib.load(MODEL_PATH)

st.title("Wine Quality Prediction")
st.write("Enter the chemical attributes of the wine sample to predict whether it is **Good (≥7)** or **Not Good (<7)**.")


fixed_acidity = st.number_input("Fixed Acidity", min_value=0.0, step=0.1)
volatile_acidity = st.number_input("Volatile Acidity", min_value=0.0, step=0.01)
citric_acid = st.number_input("Citric Acid", min_value=0.0, step=0.01)
residual_sugar = st.number_input("Residual Sugar", min_value=0.0, step=0.1)
chlorides = st.number_input("Chlorides", min_value=0.0, step=0.001)
free_sulfur_dioxide = st.number_input("Free Sulfur Dioxide", min_value=0.0, step=1.0)
total_sulfur_dioxide = st.number_input("Total Sulfur Dioxide", min_value=0.0, step=1.0)
density = st.number_input("Density", min_value=0.0, step=0.0001, format="%.5f")
pH = st.number_input("pH", min_value=0.0, step=0.01)
sulphates = st.number_input("Sulphates", min_value=0.0, step=0.01)
alcohol = st.number_input("Alcohol", min_value=0.0, step=0.1)


if st.button("Predict Wine Quality"):
  
    input_data = pd.DataFrame([{
        "fixed acidity": fixed_acidity,
        "volatile acidity": volatile_acidity,
        "citric acid": citric_acid,
        "residual sugar": residual_sugar,
        "chlorides": chlorides,
        "free sulfur dioxide": free_sulfur_dioxide,
        "total sulfur dioxide": total_sulfur_dioxide,
        "density": density,
        "pH": pH,
        "sulphates": sulphates,
        "alcohol": alcohol
    }])


    proba = model.predict_proba(input_data)[0,1]


    threshold = 0.4
    prediction = 1 if proba >= threshold else 0

    if prediction == 1:
        st.success(f"✅ This wine is predicted to be GOOD (≥7). Confidence: {proba:.2f}")
    else:
        st.error(f"❌ This wine is predicted to be NOT GOOD (<7). Confidence: {1-proba:.2f}")
