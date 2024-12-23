import streamlit as st
import pandas as pd
import pickle  # For loading saved models
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

filename = 'qda_model.sav'
class_header = 'class_header.jpg'
qda_model = pickle.load(open(filename, 'rb'))
df = pd.read_csv('USDI.csv')

# Define the app
st.title("Binary Classification with QDA")

image = Image.open(class_header)
st.image(image, use_container_width=True)


with st.expander('Data'):
    st.write('**Raw Data**')
    df2 = pd.read_csv('Trend.csv')
    df2

# Sidebar: Options for Users
st.sidebar.header("Choose a View")
options = st.sidebar.radio("Choose the type of chart:", ["Trends", "Price & Volume", "Correlation Matrix"])
trend_cols = ['USB_Trend', 'OF_Trend', 'OS_Trend', 'PLD_Trend', 'SF_Trend', 'PLT_Trend', 'EU_Trend']
# 1. **Visualize Trends** (based on the trend variables)
if options == "Trends":
    st.header("Trend Variables Over Time")

    # Convert trend columns to numeric (if not already)
    trend_data = df2[trend_cols].apply(pd.to_numeric, errors='coerce')

    # Debugging: Check the data types after conversion
    st.write("Trend Data Types:")
    st.write(trend_data.dtypes)
    
    # Debugging: Check the first few rows of the cleaned data
    st.write("Cleaned Trend Data Sample:")
    st.write(trend_data.head())

    # Ensure trend_data is not empty
    if trend_data.empty:
        st.error("The trend data is empty after cleaning (NaN/infinite values removed).")
    else:
        # Check for NaN values in the trend data (after conversion)
        nan_count = trend_data.isna().sum()
        st.write(f"NaN values in each column:\n{nan_count}")
        
        # If there are NaN values, handle them by filling with the mean (or another method)
        if nan_count.any():
            trend_data = trend_data.fillna(trend_data.mean())
            st.write("NaN values were filled with the mean of each column.")
        
        # Plot the trends over time for each trend variable
        try:
            st.subheader("Trend Variables Plot")
            
            # Create the figure and axis explicitly
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Plot the trend data
            trend_data.plot(kind='line', marker='o', ax=ax)
            
            # Customize the plot (title, labels, etc.)
            ax.set_title("Trend Variables Over Time")
            ax.set_xlabel("Index")
            ax.set_ylabel("Trend Value")
            
            # Display the plot in the Streamlit app
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Error while plotting: {str(e)}")

# 2. **Visualize USDI Price and Volume** 
elif options == "Price & Volume":
    st.header("USDI Price and Volume Analysis")

    st.subheader("USDI Price Over Time")
    plt.figure(figsize=(10, 6))
    plt.plot(df['USDI_Price'], label="USDI Price", marker='o')
    plt.title("USDI Price Over Time")
    plt.xlabel("Index")
    plt.ylabel("Price (USD)")
    plt.legend()
    st.pyplot()

    st.subheader("USDI Volume Over Time")
    plt.figure(figsize=(10, 6))
    plt.plot(df['USDI_Volume'], label="USDI Volume", marker='o', color='orange')
    plt.title("USDI Volume Over Time")
    plt.xlabel("Index")
    plt.ylabel("Volume")
    plt.legend()
    st.pyplot()

    # Scatter plot of price vs. volume
    st.subheader("Price vs. Volume Scatter Plot")
    plt.figure(figsize=(8, 6))
    plt.scatter(df['USDI_Price'], df['USDI_Volume'], alpha=0.7, color='green')
    plt.title("Price vs Volume")
    plt.xlabel("Price (USD)")
    plt.ylabel("Volume")
    st.pyplot()

# 3. **Correlation Matrix** (For trends & other variables)
elif options == "Correlation Matrix":
    st.header("Correlation Between Variables")
    
    # Select numerical columns for correlation
    correlation_cols = ['USDI_Price', 'USDI_Open', 'USDI_High', 'USDI_Low', 'USDI_Volume']
    correlation_data = df[correlation_cols]
    
    # Calculate and display correlation matrix
    corr_matrix = correlation_data.corr()
    st.write("Correlation Matrix:")
    st.write(corr_matrix)
    
    # Plot Correlation Heatmap
    st.subheader("Correlation Heatmap")
    plt.figure(figsize=(10, 6))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=1)
    st.pyplot()


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
