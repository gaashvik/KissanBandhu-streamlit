import streamlit as st
import pandas as pd
import joblib
from streamlit_lottie import st_lottie
import requests
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    else:
        return r.json()

model_filename = "fp.joblib"
model = joblib.load(model_filename)

# Label Encoders for Soil Type, Crop Type, and Fertilizer Name
le_soil = joblib.load("le_soil.joblib")  # Load the saved LabelEncoder for Soil Type
le_crop = joblib.load("le_crop.joblib")  # Load the saved LabelEncoder for Crop Type
le_fertilizer = joblib.load("le_fertilizer.joblib")  # Load the saved LabelEncoder for Fertilizer Name

# Function to preprocess user input and make predictions
def predict_fertilizer(temperature, humidity, moisture, soil_type, crop_type, nitrogen, potassium, phosphorous):
    # Preprocess user input
    user_input = pd.DataFrame([[temperature, humidity, moisture, soil_type, crop_type, nitrogen, potassium, phosphorous]])
    user_input[3] = le_soil.transform(user_input[3])
    user_input[4] = le_crop.transform(user_input[4])

    # Make prediction
    prediction = model.predict(user_input)
    return le_fertilizer.inverse_transform(prediction)[0]

rty, rtt = st.columns([3, 2])

with rty:
    original_title = '<p style="font-family:American Captain; color:#038E6A; -webkit-text-stroke: 2px black; font-size: 70px;">Fertilizer Perdiction</p>'
    st.markdown(original_title, unsafe_allow_html=True)

with rtt:
    lottie_url = "https://lottie.host/6efab570-9f79-4866-b0ee-05a40c37f9e4/NZqya3KtMx.json"
    you = load_lottieurl(lottie_url)
    st_lottie(you, speed=1, loop=True, quality="medium")
# User input
temperature = st.text_input("Temperature", 25)
humidity = st.text_input("Humidity", 50)
moisture = st.text_input("Moisture", 50)
soil_type = st.selectbox("Soil Type", ["Sandy", "Loamy", "Black", "Red", "Clayey"])
crop_type = st.selectbox("Crop Type", ["Maize", "Sugarcane", "Cotton", "Tobacco", "Paddy", "Barley", "Oil seeds", "Wheat", "Millets"])
nitrogen = st.text_input("Nitrogen", 50)
potassium = st.text_input("Potassium", 50)
phosphorous = st.text_input("Phosphorous", 50)

# Make prediction on user input
if st.button("Predict"):
    prediction = predict_fertilizer(int(temperature), int(humidity), int(moisture), soil_type, crop_type, int(nitrogen), int(potassium), int(phosphorous))
    st.success(f"The predicted fertilizer is: {prediction}")
