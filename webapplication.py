import streamlit as st
from PIL import Image
import tensorflow as tf
from tensorflow import keras
import numpy as np

# Load your trained skin cancer detection model
model = keras.models.load_model('C:/Users/Ajit Kumar Biswas/Desktop/skin_cancer_detetction/skin_cancer_detection_model_transfer_learning_finetuned_best.h5')

# Define a function to make predictions
def detect_skin_cancer(image):
    # Preprocess the image (resize, normalize, etc.)
    image = np.asarray(image) / 255.0  # Normalize to [0, 1]
    image = tf.image.resize(image, (299, 299))
    image = np.expand_dims(image, axis=0)  # Add batch dimension

    # Make a prediction
    prediction = model.predict(image)

    # Return the result
    return prediction[0][0]

# Streamlit web app
st.title('Skin Cancer Detection App')
st.write('Welcome to the Skin Cancer Detection App. Upload an image and we will analyze it for potential skin cancer.')

uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    if st.button('Detect Skin Cancer'):
        result = detect_skin_cancer(image)
        confidence = result * 100  # Convert result to percentage

        if result > 0.5:
            st.error(f'Prediction: Melanoma (Confidence: {confidence:.2f}%)')
            st.write("It's recommended to consult a dermatologist for further evaluation.")
        else:
            st.success(f'Prediction: Non-Melanoma (Confidence: {100 - confidence:.2f}%)')
            st.write("No immediate concerns detected. Regular skin checks are advised for preventive care.")

