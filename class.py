import streamlit as st
import pandas as pd
import pickle  # For loading saved models
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go

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
    st.header("Interactive Relationship Diagram for Trend Variables")

    # Let the user select two variables to compare
    variable1 = st.selectbox("Select the first trend variable:", df2.columns)
    variable2 = st.selectbox("Select the second trend variable:", df2.columns)

    # Check if user selects the same variable for both
    if variable1 == variable2:
        st.error("Please select two different variables to compare.")
    else:
        # Show the relationship between the two selected variables
        st.subheader(f"Relationship between {variable1} and {variable2}")

        # Plot the scatter plot using Plotly
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=df2[variable1],
            y=df2[variable2],
            mode='markers',
            marker=dict(size=10, color='blue', opacity=0.7),
            text=df2.index,  # Hover information (index of the data points)
            name=f'{variable1} vs {variable2}'
        ))

        fig.update_layout(
            title=f"Relationship between {variable1} and {variable2}",
            xaxis_title=variable1,
            yaxis_title=variable2,
            hovermode="closest",
            template="plotly_dark"
        )

        # Display the plot
        st.plotly_chart(fig)

        # Optionally, show the correlation coefficient between the selected variables
        correlation = df2[variable1].corr(df2[variable2])
        st.write(f"Correlation between {variable1} and {variable2}: {correlation:.2f}")

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
