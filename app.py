import streamlit as st
import pandas as pd
import pickle

# Load model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Load dataset (used for median values and dropdown)
df = pd.read_csv("housing.csv")

st.set_page_config(page_title="House Price Prediction", page_icon="🏠")

st.title("🏠 California House Price Prediction")

st.write("Enter the house details below:")

# User Inputs
longitude = st.number_input("Longitude", value=float(df["longitude"].median()))
latitude = st.number_input("Latitude", value=float(df["latitude"].median()))
housing_median_age = st.number_input("Housing Median Age", value=int(df["housing_median_age"].median()))
total_rooms = st.number_input("Total Rooms", value=int(df["total_rooms"].median()))
total_bedrooms = st.number_input("Total Bedrooms", value=int(df["total_bedrooms"].median()))
population = st.number_input("Population", value=int(df["population"].median()))
households = st.number_input("Households", value=int(df["households"].median()))
median_income = st.number_input("Median Income", value=float(df["median_income"].median()))

ocean_proximity = st.selectbox(
    "Ocean Proximity",
    df["ocean_proximity"].unique()
)

# Prediction
if st.button("Predict House Price"):

    input_data = pd.DataFrame({
        "longitude": [longitude],
        "latitude": [latitude],
        "housing_median_age": [housing_median_age],
        "total_rooms": [total_rooms],
        "total_bedrooms": [total_bedrooms],
        "population": [population],
        "households": [households],
        "median_income": [median_income],
        "ocean_proximity": [ocean_proximity]
    })

    # One-hot encoding
    input_data = pd.get_dummies(input_data, columns=["ocean_proximity"], drop_first=True)

    # Prepare training columns
    train_data = df.drop("median_house_value", axis=1)
    train_data["total_bedrooms"] = train_data["total_bedrooms"].fillna(train_data["total_bedrooms"].median())
    train_data = pd.get_dummies(train_data, columns=["ocean_proximity"], drop_first=True)

    # Match columns
    input_data = input_data.reindex(columns=train_data.columns, fill_value=0)

    prediction = model.predict(input_data)[0]

    st.success(f"Predicted House Price: ${prediction:,.2f}")