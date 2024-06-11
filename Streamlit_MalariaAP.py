import streamlit as st
import gdown
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from PIL import Image

# Define the Google Drive file ID
file_id = '1jyQqApyt9zSKiHPk-zDiXituTxajV8Bn'
model_path = 'vgg16_model.h5'

# Download the model from Google Drive if it does not exist locally
if not os.path.exists(model_path):
    gdown.download(f'https://drive.google.com/uc?export=download&id={file_id}', model_path, quiet=False)

# Confirm file download
print(f"File exists: {os.path.exists(model_path)}")

# Load the model
model = tf.keras.models.load_model(model_path)

def predict_malaria(img):
    img = img.resize((224, 224))
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
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Image', use_column_width=True)
    st.write("")
    st.write("Classifying...")
    prediction = predict_malaria(img)
    if prediction[0][0] > 0.5:
        st.write("The cell is **infected** by malaria.")
    else:
        st.write("The cell is **not infected** by malaria.")






