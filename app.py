# streamlit_app.py
import streamlit as st
import pickle
import numpy as np

# Load models
ridge_model = pickle.load(open('models/ridge.pkl', 'rb'))
standard_scaler = pickle.load(open('models/scaler.pkl', 'rb'))

# Streamlit App
st.title("🔥 Forest Fire Prediction App")

# Input fields
Temperature = st.number_input("Temperature (°C)", step=0.1)
RH = st.number_input("Relative Humidity (%)", step=0.1)
Ws = st.number_input("Wind Speed (km/h)", step=0.1)
Rain = st.number_input("Rain (mm)", step=0.1)
FFMC = st.number_input("FFMC Index", step=0.1)
DMC = st.number_input("DMC Index", step=0.1)
ISI = st.number_input("ISI Index", step=0.1)
Classes = st.selectbox("Fire Class (0 = low, 1 = high)", [0, 1])
Region = st.selectbox("Region (1 = Bejaia, 2 = Sidi-Bel Abbes)", [1, 2])

# Predict button
if st.button("Predict Fire Area"):
    input_data = np.array([[Temperature, RH, Ws, Rain, FFMC, DMC, ISI, Classes, Region]])
    scaled_data = standard_scaler.transform(input_data)
    result = ridge_model.predict(scaled_data)
    st.success(f"🔥 Predicted Burned Area: {result[0]:.2f} ha")
