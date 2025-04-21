import streamlit as st
import tensorflow as tf

@st.cache_resource
def load_model():
  model=tf.keras.models.load_model('chest_xray_model.hdf5')
  return model
model=load_model()
st.write("""
# Pneumonia Detection"""
)
file=st.file_uploader("Choose xray photo from computer",type=["jpg","png"])

import cv2
from PIL import Image,ImageOps
import numpy as np
def import_and_predict(image_data,model):
    size=(64,64)
    image=ImageOps.fit(image_data,size)
    img=np.asarray(image)
    img_reshape=img[np.newaxis,...]
    prediction=model.predict(img_reshape)
    return prediction
if file is None:
    st.text("Please upload an image file")
else:
    image=Image.open(file)
    st.image(image,use_column_width=True)
    prediction=import_and_predict(image,model)
    class_names=['NORMAL, PNEUMONIA']
    string="OUTPUT : "+class_names[np.argmax(prediction)]
    st.success(string)
