from PIL import Image
import streamlit as st
from models import predict
from views import formatDisplay
import time

st.title("Plant Disease Classification")
st.caption("Apple, Bell pepper, Blueberry, Cherry, Corn, Peach, Potato, Raspberry, Soyabean, Squash, Strawberry, Tomato, Grape")
st.divider()

fup = st.file_uploader("Upload a leaf image", type = "jpg", accept_multiple_files=False)

col1, col2 = st.columns([3, 2])

with col1:
    if fup is not None:
        img = Image.open(fup)
        st.image(img, caption = 'Uploaded Image', use_column_width = True)

with col2:
    if fup is not None:
        time.sleep(0.5)
        st.write("Prediction:")
        with st.spinner('Wait a seconds ...'):
            predictions = predict(fup)
            
            for prediction in predictions:
                st.success(formatDisplay(prediction))