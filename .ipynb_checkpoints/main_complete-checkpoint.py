import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os

# Load
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = joblib.load(os.path.join(BASE_DIR, "model.pkl"))
mae = joblib.load(os.path.join(BASE_DIR, "mae.pkl"))

# UI
st.set_page_config(page_title="House Price Predictor", layout="centered")

st.title("🏠 Delhi House Price Prediction")
st.markdown("Fill the details below to estimate house price")

# Input
area = st.number_input("Area (sqft)", min_value=0.0)
bedrooms = st.number_input("Bedrooms", min_value=0, step=1)
bathrooms = st.number_input("Bathrooms", min_value=0, step=1)
balcony = st.number_input("Balcony", min_value=0, step=1)
parking = st.number_input("Parking", min_value=0, step=1)

status = st.selectbox("Status", ["Ready_to_move", "Under_Construction"])

furnished = st.selectbox(
    "Furnished Status",
    ["Furnished", "Semi-Furnished", "Unfurnished"]
)

building_type = st.selectbox(
    "Type of Building",
    ["Flat", "Builder_Floor"]
)

neworold = st.selectbox(
    "Property Type",
    ["New Property", "Resale"]
)

location_cluster = st.selectbox("Location Cluster", [0, 1, 2, 3])

# Predict
if st.button("Predict"):

    # ✅ Validation
    if area <= 0:
        st.error("❌ Area must be greater than 0")
    
    else:
        with st.spinner("Predicting price..."):

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
            pred_actual = np.exp(pred_log)[0]

            lower = max(0, pred_actual - mae)
            upper = pred_actual + mae

        # Output ji
        st.success(f"🏠 Estimated Price: ₹ {pred_actual:,.0f}")
        st.info(f"📊 Price Range: ₹ {lower:,.0f} - ₹ {upper:,.0f}")
        st.caption(f"⚠️ Model Error: ± ₹ {mae:,.0f}")