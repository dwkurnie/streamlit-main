import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load model
@st.cache_resource
def load_model():
    return joblib.load("XGB-Model1.pkl")

model = load_model()

st.title("Prediksi Revenue E-Commerce")

st.write("Masukkan nilai untuk setiap fitur:")

# === Input Features ===
Administrative = st.number_input("Administrative", min_value=0, value=0)
Administrative_Duration = st.number_input("Administrative_Duration", min_value=0.0, value=0.0)
Informational = st.number_input("Informational", min_value=0, value=0)
Informational_Duration = st.number_input("Informational_Duration", min_value=0.0, value=0.0)
ProductRelated = st.number_input("ProductRelated", min_value=0, value=0)
ProductRelated_Duration = st.number_input("ProductRelated_Duration", min_value=0.0, value=0.0)
BounceRates = st.number_input("BounceRates", min_value=0.0, max_value=1.0, value=0.0)
ExitRates = st.number_input("ExitRates", min_value=0.0, max_value=1.0, value=0.0)
PageValues = st.number_input("PageValues", min_value=0.0, value=0.0)
SpecialDay = st.number_input("SpecialDay", min_value=0.0, max_value=1.0, value=0.0)

Month = st.selectbox("Month", ["Jan","Feb","Mar","Apr","May","June","Jul","Aug","Sep","Oct","Nov","Dec"])
OperatingSystems = st.number_input("OperatingSystems", min_value=1, max_value=8, value=1)
Browser = st.number_input("Browser", min_value=1, max_value=13, value=1)
Region = st.number_input("Region", min_value=1, max_value=9, value=1)
TrafficType = st.number_input("TrafficType", min_value=1, max_value=20, value=1)
VisitorType = st.selectbox("VisitorType", ["Returning_Visitor", "New_Visitor", "Other"])
Weekend = st.selectbox("Weekend", ["True", "False"])

# Encode categorical features
Month_map = {'Jan':1,'Feb':2,'Mar':3,'Apr':4,'May':5,'June':6,'Jul':7,'Aug':8,'Sep':9,'Oct':10,'Nov':11,'Dec':12}
VisitorType_map = {"Returning_Visitor":1,"New_Visitor":0,"Other":0}
Weekend_map = {"True":1,"False":0}

Month_encoded = Month_map[Month]
VisitorType_encoded = VisitorType_map[VisitorType]
Weekend_encoded = Weekend_map[Weekend]

# Button untuk prediksi
if st.button("Prediksi Revenue"):
    features = np.array([[Administrative, Administrative_Duration, Informational, Informational_Duration,
                          ProductRelated, ProductRelated_Duration, BounceRates, ExitRates, PageValues,
                          SpecialDay, Month_encoded, OperatingSystems, Browser, Region, TrafficType,
                          VisitorType_encoded, Weekend_encoded]])

    prediction = model.predict(features)
result = "Yes" if prediction[0] == 1 else "No"

# Tentukan kata "tidak" hanya jika hasil = No
tidak = "" if result == "Yes" else "tidak"

st.success(f"Hasil Prediksi Revenue: {result}")
st.write(f"Konsumen berkemungkinan besar untuk {tidak} melanjutkan pembelian.")


