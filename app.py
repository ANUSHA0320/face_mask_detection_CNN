import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Load the model without compiling
model = tf.keras.models.load_model('mask_bbox_model.h5', compile=False)

# Compile the model with the correct loss
model.compile(optimizer='adam', loss='mse')

st.title("Face Mask Detection")

uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # Preprocess the image
    img = image.resize((128, 128))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Predict
    prediction = model.predict(img_array)[0][0]
    label = "With Mask" if prediction < 0.5 else "Without Mask"
    st.write(f"Prediction: **{label}**")