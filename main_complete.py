import streamlit as st
import joblib
import pandas as pd
import numpy as np

#model = joblib.load("model.pkl")
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "model.pkl")

model = joblib.load(model_path)

st.title("House Price Prediction")

area = st.number_input("Area")
bedrooms = st.number_input("Bedrooms", step=1)
bathrooms = st.number_input("Bathrooms", step=1)
balcony = st.number_input("Balcony", step=1)
parking = st.number_input("Parking", step=1)

status = st.selectbox("Status", ["Ready_to_move", "Under_Construction"])

furnished = st.selectbox("Furnished Status",
                         ["Furnished", "Semi-Furnished", "Unfurnished"])

building_type = st.selectbox("Type of Building",
                             ["Flat", "Builder_Floor"])

neworold = st.selectbox("Property Type",
                        ["New Property", "Resale"])

location_cluster = st.selectbox("Location Cluster", [0, 1, 2, 3])

if st.button("Predict"):
    input_data = pd.DataFrame({
        "area": [area],
        "Bedrooms": [bedrooms],
        "Bathrooms": [bathrooms],
        "Balcony": [balcony],
        "parking": [parking],
        "Status": [status],
        "Furnished_status": [furnished],
        "type_of_building": [building_type],
        "neworold": [neworold],
        "location_cluster": [location_cluster]
    })

    pred_log = model.predict(input_data)
    pred_actual = np.exp(pred_log)

    st.success(f"Predicted Price: ₹ {round(pred_actual[0],2)}")
