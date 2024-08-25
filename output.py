import streamlit as st
import cv2
import numpy as np
import joblib

# Load the saved model
model = joblib.load('svm_trashnet_model.pkl')

def preprocess_image(image):
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    img = cv2.resize(img, (128, 128))
    img = img.flatten()
    return img.reshape(1, -1)

st.title("TrashNet Classifier")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
if uploaded_file is not None:
    image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)
    st.image(image, channels="RGB")

    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)

    st.write(f"Prediction: {prediction[0]}")
