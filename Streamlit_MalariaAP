import streamlit as st
import gdown
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# Define the Google Drive file ID
file_id = '1kTCFJfh754rhkE9LR6GVM8S4-HsloOKs'
model_path = 'vgg16_model.h5'

# Download the model from Google Drive if it does not exist locally
if not os.path.exists(model_path):
    gdown.download(f'https://drive.google.com/uc?export=download&id={file_id}', model_path, quiet=False)

# Load the model
model = tf.keras.models.load_model(model_path)

def predict_malaria(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    prediction = model.predict(img_array)
    return prediction

# Streamlit app
st.title('Malaria Detection App')
st.write('Upload a photo of a blood cell to see if it is infected by malaria.')

uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
    st.write("")
    st.write("Classifying...")
    prediction = predict_malaria(uploaded_file)
    if prediction[0][0] > 0.5:
        st.write("The cell is **infected** by malaria.")
    else:
        st.write("The cell is **not infected** by malaria.")

