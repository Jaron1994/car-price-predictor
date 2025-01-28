import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import os

# ✅ Load dataset
file_path = "/Users/aarons25/Downloads/Car Price Data Doc.csv"

if not os.path.exists(file_path):
    st.error("❌ File not found! Ensure 'Car Price Data Doc.csv' is in your Downloads folder.")
    st.stop()

df = pd.read_csv(file_path)

# ✅ Data Cleaning: Ensure no missing or incorrect values
df.dropna(inplace=True)

# ✅ Select Features for Prediction (Removed "Owner_Count")
features = ["Brand", "Year", "Mileage", "Transmission"]
target = "Price"

# ✅ Encode Categorical Features (One-Hot Encoding)
encoder = OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore')
encoded_data = encoder.fit_transform(df[['Brand', 'Transmission']])

# Convert to DataFrame
encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(['Brand', 'Transmission']))

# Concatenate encoded categorical features with numerical features
df_encoded = pd.concat([df[["Year", "Mileage", "Price"]].reset_index(drop=True), encoded_df], axis=1)

# ✅ Prepare Features (X) and Target Variable (y)
X = df_encoded.drop(columns=["Price"])
y = df_encoded["Price"]

# ✅ Train the Linear Regression Model
model = LinearRegression()
model.fit(X, y)

# ✅ Streamlit UI
st.title("🚗 Car Price Predictor")
st.write("Predict the estimated price of a car based on its **Brand, Year, Mileage, and Transmission.**")

# 🔹 Step 1: Select Brand
selected_brand = st.selectbox("🚘 Select Car Brand", sorted(df["Brand"].unique()))

# 🔹 Step 2: Select Transmission (Dynamically Filtered)
filtered_transmissions = df[df["Brand"] == selected_brand]["Transmission"].unique()
selected_transmission = st.selectbox("⚙️ Select Transmission", sorted(filtered_transmissions))

# 🔹 Step 3: Select Year and Mileage
year = st.slider("📅 Select Car Year", int(df["Year"].min()), int(df["Year"].max()), 2015)
mileage = st.number_input("🚗 Enter Mileage (miles)", int(df["Mileage"].min()), int(df["Mileage"].max()), 50000)

# ✅ Encode User Input
user_input = pd.DataFrame([[year, mileage]], columns=["Year", "Mileage"])

# Ensure all encoded features exist in user_input
for col in encoder.get_feature_names_out(['Brand', 'Transmission']):
    user_input[col] = 0

# Set the selected values to 1 (matching the selection)
brand_column = f"Brand_{selected_brand}" if f"Brand_{selected_brand}" in user_input.columns else None
transmission_column = f"Transmission_{selected_transmission}" if f"Transmission_{selected_transmission}" in user_input.columns else None

for col in [brand_column, transmission_column]:
    if col:
        user_input[col] = 1

# ✅ Ensure user_input has all features needed for prediction
missing_features = set(X.columns) - set(user_input.columns)
for feature in missing_features:
    user_input[feature] = 0  # Default missing features to 0

# ✅ Predict Price
predicted_price = model.predict(user_input[X.columns])

# ✅ Display Prediction
st.success(f"💰 **Estimated Car Price: ${predicted_price[0]:,.2f}**")
