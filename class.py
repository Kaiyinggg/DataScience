import streamlit as st
import pandas as pd
import pickle  # For loading saved models
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px


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


# if options == "Trends":
#     st.header("Binary Trend Variables Stacked Bar Plot")

#     # Show the list of trend variables
#     st.write("Trend Variables:", df2.columns)

#     # Create a stacked bar plot for pairwise combinations
#     # Pivot the data for binary trend pairs
#     pairwise_counts = pd.crosstab(df2['USDI_Trend'], df2['EU_Trend'])

#     # Plot as stacked bar chart
#     fig = px.bar(pairwise_counts, 
#                  barmode='stack', 
#                  labels={'USB_Trend': 'USDI Trend', 'EU_Trend': 'OF Trend'},
#                  title="Stacked Bar Plot of USDI_Trend vs EU_Trend")

#     # Display the plot
#     st.plotly_chart(fig)

if options == "Trends":
    st.header("Trend Variables Analysis")
    # Let the user select two variables to compare
    variable1 = st.selectbox("Select the first trend variable:", df2.columns)
    variable2 = st.selectbox("Select the second trend variable:", df2.columns)

    # Check if user selects the same variable for both
    if variable1 == variable2:
        st.error("Please select two different variables to compare.")
    else:
        # Show the relationship between the two selected variables
        
        st.subheader(f"Stacked Bar Plot relationship between {variable1} and {variable2}")

        pairwise_counts = pd.crosstab(df2[variable1], df2[variable2])

        # Plot the data as a stacked bar chart
        fig = px.bar(pairwise_counts, 
                     barmode='stack', 
                     labels={variable1: variable1, variable2: variable2},
                     title=f"Stacked Bar Plot of {variable1} vs {variable2}")
    
        # Display the plot
        st.plotly_chart(fig)
        pairwise_counts = pd.crosstab(df2[variable1], df2[variable2])


        
        # Prepare the data for the line plot
        data_for_line_plot = pairwise_counts.stack().reset_index(name="Count")
        
        # Generate the x-axis labels for the combinations (0, 0), (0, 1), (1, 0), (1, 1)
        data_for_line_plot['Combination'] = data_for_line_plot.apply(
            lambda row: f"({row[variable1]},{row[variable2]})", axis=1)
            #lambda row: f"({variable1}={row[variable1]}, {variable2}={row[variable2]})", axis=1)
        
        # Sort by combination for plotting
        data_for_line_plot = data_for_line_plot.sort_values(by="Combination")
        
        # Create a line plot with Plotly Graph Objects
        fig_line = go.Figure()
    
        fig_line.add_trace(go.Scatter(x=data_for_line_plot['Combination'], 
                                     y=data_for_line_plot['Count'], 
                                     mode='lines+markers', 
                                     name='Combination Counts',
                                     line=dict(width=2),
                                     text=data_for_line_plot['Combination'], 
                                     hoverinfo='text'))
    
        # Update the layout for better presentation
        fig_line.update_layout(
            title=f"Line Graph of Counts for {variable1} vs {variable2}",
            xaxis_title="Binary Combination",
            yaxis_title="Count",
            showlegend=True
        )
    
        # Display the line graph
        st.plotly_chart(fig_line)

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
