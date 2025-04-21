import streamlit as st
import tensorflow as tf

@st.cache_resource
def load_model():
  model=tf.keras.models.load_model('chest_xray_model.h5')
  return model
model=load_model()
st.write("""
# Pneumonia Detection"""
)
file=st.file_uploader("Choose xray photo from computer",type=["jpg","png"])

import cv2
from PIL import Image,ImageOps
import numpy as np
def import_and_predict(image_data, model):
    size = (150, 150)
    image = ImageOps.fit(image_data, size, Image.Resampling.LANCZOS)
    img = np.asarray(image)
    img_reshape = img[np.newaxis, ...]
    prediction = model.predict(img_reshape)
    return prediction

model = load_model()
file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file).convert('RGB')
    st.image(image, use_column_width=True)
    prediction = import_and_predict(image, model)
    class_names = ['Normal', 'Pneumonia']
    predicted_class = class_names[np.argmax(prediction)]
    confidence = f"{np.max(prediction) * 100:.2f}%"
    string = f"OUTPUT : {predicted_class} (Confidence: {confidence})"
    if predicted_class == 'Pneumonia':
        st.error(string)
    else:
        st.success(string)
