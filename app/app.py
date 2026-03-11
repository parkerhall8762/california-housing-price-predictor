import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("../models/housing_model.pkl")

st.title("California Housing Price Predictor")

st.write("Enter housing information to predict the house price.")

median_income = st.number_input("Median Income", min_value=0.0)
housing_age = st.number_input("Housing Median Age", min_value=0.0)
total_rooms = st.number_input("Total Rooms", min_value=0.0)
population = st.number_input("Population", min_value=0.0)

if st.button("Predict Price"):
    
    data = pd.DataFrame({
        "median_income": [median_income],
        "housing_median_age": [housing_age],
        "total_rooms": [total_rooms],
        "population": [population]
    })

    prediction = model.predict(data)

    st.success(f"Estimated House Price: ${prediction[0]:,.2f}")
