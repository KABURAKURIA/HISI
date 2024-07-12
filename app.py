import streamlit as st
import numpy as np
from PIL import Image, ImageOps
from tensorflow.keras.models import load_model
from streamlit_lottie import st_lottie
import requests

# Load the model and labels
model = load_model("keras_model.h5", compile=False)
class_names = open("labels.txt", "r").readlines()

# Function to preprocess the image
def preprocess_image(image):
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array
    return data

# Function to predict emotion
def predict_emotion(image):
    data = preprocess_image(image)
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]
    return class_name[2:].strip(), confidence_score

# Function to load Lottie animation
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Load a Lottie animation
lottie_animation = load_lottieurl("https://assets2.lottiefiles.com/packages/lf20_yadoe3im.json")

# Streamlit app
st.title("HISI - Emotion Detection")

# Image upload and camera capture options
image_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
camera_image = st.camera_input("Or take a picture")

if image_file is not None or camera_image is not None:
    if image_file is not None:
        image = Image.open(image_file).convert("RGB")
    else:
        image = Image.open(camera_image).convert("RGB")

    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Predict Emotion"):
        with st.spinner('Analyzing...'):
            # Display Lottie animation
            animation_placeholder = st.empty()
            with animation_placeholder.container():
                st_lottie(lottie_animation, height=200, width=200)

            class_name, confidence_score = predict_emotion(image)

            # Remove the animation after processing
            animation_placeholder.empty()

            st.write(f"Class: {class_name}")
            st.write(f"Confidence Score: {confidence_score:.2f}")
