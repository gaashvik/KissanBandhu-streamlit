import streamlit as st
import pandas as pd
import requests
import numpy as np
import joblib
from streamlit_lottie import st_lottie

def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    else:
        return r.json()

# Load the crop descriptions from the CSV file
crop_data = pd.read_csv('crop_rec.csv')

# Rest of your Streamlit code
rty, rtt = st.columns([3, 2])

with rty:
    original_title = '<p style="font-family:American Captain; color:#038E6A; -webkit-text-stroke: 2px black; font-size: 50px;">Crop Recommendation</p>'
    st.markdown(original_title, unsafe_allow_html=True)

with rtt:
    lottie_url = "https://lottie.host/36508d6a-1247-4cd4-bd2d-ab594462e2d8/pWA8egbuHm.json"
    you = load_lottieurl(lottie_url)
    st_lottie(you, speed=1, loop=True, quality="medium")

# Get user input for each feature
nitrogen = st.slider("Nitrogen", min_value=0, max_value=100, value=50)
phosphorus = st.slider("Phosphorus", min_value=0, max_value=100, value=50)
potassium = st.slider("Potassium", min_value=0, max_value=100, value=50)
temp = st.slider("Temperature", min_value=0, max_value=100, value=50)
humidity = st.slider("Humidity", min_value=0, max_value=100, value=50)
ph = st.slider("pH", min_value=0, max_value=14, value=7)
rainfall = st.slider("Rainfall", min_value=0, max_value=100, value=50)
submit = st.button("Submit")

if submit:
    user_input = pd.DataFrame({
        'Nitrogen': [nitrogen],
        'Phosphorus': [phosphorus],
        'Potassium': [potassium],
        'Temp': [temp],
        'Humidity': [humidity],
        'pH': [ph],
        'Rainfall': [rainfall]
    })

    loaded_model = joblib.load('crop_rec.joblib')
    st.write(user_input)
    prediction = loaded_model.predict(user_input.values.reshape(1, -1))
    st.subheader("Based on Your Input Our Recommendation is-:")
    q1,q2=st.columns([3,2])
    with q1:
        st.write("")
        st.write("")
        st.write("")
        st.header(f"{prediction[0]}")
    with q2:
        lottie_url ="https://lottie.host/7eb75c24-749a-4f7e-9eb1-4364359a64b8/cLxJUDLsBx.json"
        you= load_lottieurl(lottie_url)
        st_lottie(
            you,
            speed=1,
            loop=True,
            quality="medium",
            )
    crop_description = crop_data[crop_data['class_name'] == prediction[0]]['description'].values[0]
    st.write("***Crop Description***:",crop_description)
