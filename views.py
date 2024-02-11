import streamlit as st
import string

def formatDisplay(prediction):
    class_name = prediction[0]
    plant_name = ' '.join((class_name.split(' ')[0]).split('_'))
    disease_name = ' '.join(class_name.split(' ')[1:])
    prob = "{:.2f}%".format(prediction[1]*100)

    return f"**{string.capwords(plant_name)}**  \n  {string.capwords(disease_name)} [{prob}]"