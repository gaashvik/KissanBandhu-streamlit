import requests
import streamlit as st
from PIL import Image as PILImage
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as tf_image
import numpy as np
import streamlit as st
import mysql.connector
import matplotlib.pyplot as plt
from lime import lime_image
from streamlit_lottie import st_lottie

from lime.wrappers.scikit_image import SegmentationAlgorithm
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    else:
        return r.json()
def get_plant_disease_info(plant_name,disease_status):
    # Connect to the MySQL database
    db_connection = mysql.connector.connect(
        host='localhost',
        user='root',
        password='root',
        database='plant'
    )
    p_n=plant_name+"___"+disease_status
    # Query the database for information
    query = f"SELECT * FROM plant_info WHERE plant_name = '{p_n}';"
    cursor = db_connection.cursor(dictionary=True)
    cursor.execute(query)
    result = cursor.fetchone()

    # Close the database connection
    cursor.close()
    db_connection.close()

    return result

train_datagen = ImageDataGenerator(rescale=1./255)
train_dir = r"D:\downloads\archive (1)\New Plant Diseases Dataset(Augmented)\New Plant Diseases Dataset(Augmented)\train"
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

# Load the trained model
loaded_model = load_model("model_checkpoint.h5")
# def lime_explanation(image_path, model, class_names, num_samples=1000):
#     explainer = lime_image.LimeImageExplainer()

#     # Load the image and preprocess it
#     img = tf_image.load_img(image_path, target_size=(224, 224))
#     img_array = tf_image.img_to_array(img)
#     img_array = np.expand_dims(img_array, axis=0)
#     img_array /= 255.0

#     # Get the top class index
#     top_class = np.argmax(model.predict(img_array)[0])

#     # Explain the prediction
#     explanation = explainer.explain_instance(
#         img_array[0], model.predict, top_labels=1, hide_color=0, num_samples=num_samples
#     )

#     # Get the mask
#     _, mask = explanation.get_image_and_mask(
#         top_class,
#         positive_only=False,
#         num_features=5,
#         hide_rest=False,
#         min_weight=0.01  # Adjust min_weight to control the number of superpixels displayed
#     )

#     # Overlay the mask on the original image
#     superimposed_img = (img_array[0] * 255).astype(np.uint8)
#     masked_img = np.zeros_like(superimposed_img)
#     for i in range(3):  # Loop over channels (R, G, B)
#         masked_img[:, :, i] = np.where(mask[:, :] > 0, superimposed_img[:, :, i], 0)

#     # Display the original image and the image with highlighted superpixels
#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

#     ax1.imshow(superimposed_img)
#     ax1.set_title("Original Image")
#     ax1.axis("off")

#     ax2.imshow(masked_img)
#     ax2.set_title("LIME Explanation: Highlighted Superpixels")
#     ax2.axis("off")

#     plt.show()

# Function to make predictions
def make_prediction(image_path):
    img = tf_image.load_img(image_path,target_size=(224,224))
    # img = img.resize((224, 224))  # Resize the image to match the model's expected sizing
    img_array = tf_image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    prediction = loaded_model.predict(img_array)
    class_indices = train_generator.class_indices
    inv_class_indices = {v: k for k, v in class_indices.items()}
    predicted_class = inv_class_indices[np.argmax(prediction)]
    confidence = np.max(prediction)
    plant_name, disease_status = predicted_class.split('___')
    return plant_name, disease_status, confidence
rty,rtt=st.columns([3,2])
with rty:
    original_title= '<p style="font-family:American Captain; color:#038E6A; -webkit-text-stroke: 2px black; font-size: 100px;">Disease Detection</p>'
    st.markdown(original_title, unsafe_allow_html=True)
with rtt:
    lottie_url = "https://lottie.host/295e93fd-1b05-4c90-801a-6cc2a70afbd8/5fSulgE3Te.json"
    you= load_lottieurl(lottie_url)
    st_lottie(
        you,
        speed=1,
        loop=True,
        quality="medium",
        )

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type="jpg")
if uploaded_file is not None:
    col1,col2,col3 = st.columns(3)
    with col2:
        uploaded_image = PILImage.open(uploaded_file)
        st.image(uploaded_image, caption='Uploaded Image.', use_column_width=True)
        but=st.button('Predict')
    if but:
        plant_name, disease_status, confidence = make_prediction(uploaded_file)
        st.header("Here is our prediction:")
        st.subheader(f"Plant Name - {plant_name}")
        st.subheader(f"Disease Status - {disease_status}")
        st.subheader("Confidence - "+f":green[{confidence*100:.2f}]%")
        # lime_explanation(uploaded_file, loaded_model, class_names=train_generator.class_indices.values())
        plant_disease_info = get_plant_disease_info(plant_name,disease_status)
        if plant_disease_info:
            st.write(f"***Description***: {plant_disease_info['descriptions']}")
            q1,q2=st.columns([3,2])
            with q1:
                st.write("")
                st.write("")
                st.write("")
                st.write(f"***Symptoms***: {plant_disease_info['symptoms']}")
            with q2:
                lottie_url ="https://lottie.host/4749946e-6b7e-406b-ac01-a5d6338e0705/6MTMTiI8gs.json"
                you= load_lottieurl(lottie_url)
                st_lottie(
                    you,
                    speed=1,
                    loop=True,
                    quality="medium",
                    width=150,
                    height=150
                    )
            t1,t2=st.columns([1,3])
            with t1:
                lottie_url ="https://lottie.host/53dc6688-8471-4b9a-b200-020a82ebc508/PYaMIWh66e.json"
                you= load_lottieurl(lottie_url)
                st_lottie(
                    you,
                    speed=1,
                    loop=True,
                    quality="medium",
                    width=150,
                    height=150
                    )
            with t2:
                st.write("")
                st.write("")
                st.write("")
                st.write(f"***Disease Cycle***: {plant_disease_info['disease_cycle']}")
            y5,y1,y2,y4=st.columns([1,2,2,1])
            with y1:
                st.write("")
                st.write("")
                st.write("")
                st.metric(label="***:red[Danger Rate]***", value=f"{plant_disease_info['danger_rate']}")
            with y2:

                lottie_url ="https://lottie.host/df2dcd41-3dc9-4f71-92f6-8c0440b66917/HQbB6gs5Nr.json"
                you= load_lottieurl(lottie_url)
                st_lottie(
                    you,
                    speed=1,
                    loop=True,
                    quality="medium",
                    width=150,
                    height=150
                    )
                # st.info(f"Plant Information:\nDescription: {plant_disease_info['descriptions']}\nDisease Cycle: {plant_disease_info['disease_cycle']}\nSymptoms: {plant_disease_info['symptoms']}\nDanger Rate: {plant_disease_info['danger_rate']}")
        else:
            st.warning("No information available for the selected plant.")