# To run this app:
# python -m streamlit run main.py

import streamlit as st
import pandas as pd
from lgbm import model

st.title("Farmer App")

with st.form("my_form"):
    temperature = st.slider("Temperature (°C)", min_value=-10, max_value=60, value=28)
    sunlight = st.slider("Sunlight (hours/day)", min_value=0.0, max_value=24.0, value=6.0, step=0.25)
    humidity = st.slider("Humidity (%)", min_value=0, max_value=100, value=60)
    rainfall = st.slider("Rainfall (mm/year)", min_value=0, max_value=300, value=100)
    soil_ph = st.slider("Soil pH Level", min_value=0.0, max_value=14.0, value=6.5, step=0.1)
    nitrogen = st.slider("Nitrogen (N)", min_value=0, max_value=200, value=75)
    phosphorus = st.slider("Phosphorus (P)", min_value=0, max_value=200, value=35)
    potassium = st.slider("Potassium (K)", min_value=0, max_value=200, value=50)
    season = st.selectbox("Growing Season", ["Whole Year", "Kharif", "Rabi", "Zaid"])
    submitted = st.form_submit_button("Submit")
if submitted:
    feature_names = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
    X_new = pd.DataFrame([[nitrogen, phosphorus, potassium, temperature, humidity, soil_ph, rainfall]], columns=feature_names)
    newdata = model.predict(X_new)
    st.write(f"### Suggested crop : **{newdata[0].title()}**")



