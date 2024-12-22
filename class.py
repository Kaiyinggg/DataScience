import streamlit as st
import pandas as pd
import pickle  # For loading saved models
from PIL import Image



model_path = 'C:\\Users\\Asus\\Downloads\\DS\\qda_model.sav'
header_image_path = "C:\\Users\\Asus\\Downloads\\DS\\class_header.jpg"


# Load your trained model (ensure the model is saved as 'model.pkl')
with open(model_path, 'rb') as file:
    qda_model = pickle.load(file)

# Define the app
st.title("Binary Classification with QDA")

image = Image.open(header_image_path)
st.image(image, use_column_width=True)

# Input features
st.header("Input Features")
USB_Trend = st.selectbox("USB_Trend", [0, 1], help="Choose 0 or 1")
OF_Trend = st.selectbox("OF_Trend", [0, 1])
OS_Trend = st.selectbox("OS_Trend", [0, 1])
PLD_Trend = st.selectbox("PLD_Trend", [0, 1])
SF_Trend = st.selectbox("SF_Trend", [0, 1])
PLT_Trend = st.selectbox("PLT_Trend", [0, 1])
EU_Trend = st.selectbox("EU_Trend", [0, 1])

# Create a DataFrame for prediction
input_data = pd.DataFrame({
    'USB_Trend': [USB_Trend],
    'OF_Trend': [OF_Trend],
    'OS_Trend': [OS_Trend],
    'PLD_Trend': [PLD_Trend],
    'SF_Trend': [SF_Trend],
    'PLT_Trend': [PLT_Trend],
    'EU_Trend': [EU_Trend]
})

# Predict button
if st.button("Predict"):
    prediction = qda_model.predict(input_data)
    if prediction[0] == 1:
        st.success("The prediction is: Positive Trend")
    else:
        st.error("The prediction is: Negative Trend")
