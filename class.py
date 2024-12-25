import streamlit as st
import pandas as pd
import pickle  
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from streamlit_option_menu import option_menu

qda = 'qda_model.sav'
mnb = 'mnb_model.sav'
class_header = 'class_header.jpg'
trend = 'Trend.csv'
USDI = 'USDI.csv'

qda_model = pickle.load(open(qda, 'rb'))
mnb_model = pickle.load(open(mnb, 'rb'))
df = pd.read_csv(USDI)
df2 = pd.read_csv(trend)
df['Date'] = pd.to_datetime(df['Date'])

# Sidebar: Options for Users
st.sidebar.header("Menu")
# options = st.sidebar.radio("Choose the desired view:", ["Dashboard" ,"Classification", "Trends", "Price & Volume"])
# Sidebar stylish navigation
with st.sidebar:
    options = option_menu("Navigation", 
                           ["Dashboard", "Trends", "Price & Volume", "Classification"], 
                           icons=['house', 'bar-chart', 'graph-up-arrow', 'search'], 
                           menu_icon="list", 
                           default_index=0)
    
    model_options = option_menu("Model Selection", 
                           ["QDA Model", "MNB Model"], 
                           icons=['calculator', 'file-earmark-text'], 
                           menu_icon="list", 
                           default_index=0)
    

if model_options == "QDA Model":
    chosen_model = qda_model
    name = "Quadratic Discriminant Analysis"
else:
    chosen_model = mnb_model  
    name = "Multinomial Naive Bayes"
    
if options == "Dashboard":
    # Get the latest date in the dataset
    #latest_date = df["Date"].max()
    latest_date = df["Date"].iloc[-2]

    # Add a Date Picker for user to select a date
    st.title("USDI Price and Volume Metrics")
    selected_date = st.date_input("Select a Date", min_value=df["Date"].min(), max_value=df["Date"].max(), value=latest_date)

    # Find the row for the selected date
    selected_row = df[df['Date'] == pd.to_datetime(selected_date)]

    # Check if the selected date is valid (i.e., there is data for that date)
    if not selected_row.empty:
        # Get the previous day row (if available)
        previous_row = df[df['Date'] == pd.to_datetime(selected_date) - pd.Timedelta(days=1)]

        # If there is no previous row, inform the user (e.g., first data point)
        if previous_row.empty:
            st.warning("No data available for the previous day.")
            previous_row = selected_row  # Treat as the same row for first entry

        # Extract values for the selected and previous rows
        latest_price = selected_row["USDI_Price"].values[0]
        previous_price = previous_row["USDI_Price"].values[0]
        price_change = latest_price - previous_price
        price_change_pct = (price_change / previous_price) * 100 if previous_price != 0 else 0

        latest_volume = selected_row["USDI_Volume"].values[0]
        previous_volume = previous_row["USDI_Volume"].values[0]
        volume_change = latest_volume - previous_volume
        volume_change_pct = (volume_change / previous_volume) * 100 if previous_volume != 0 else 0

        latest_high = selected_row["USDI_High"].values[0]
        latest_low = selected_row["USDI_Low"].values[0]
        high_low_difference = latest_high - latest_low

        # Use columns to organize the metrics
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(label="USDI Price", value=f"${latest_price:.2f}", delta=f"{price_change:.2f} USD")

        with col2:
            st.metric(label="USDI Volume", value=f"{latest_volume:,}", delta=f"{volume_change:.2f} USD")

        with col3:
            st.metric(label="USDI High (Latest)", value=f"${latest_high:.2f}", delta=f"{high_low_difference:.2f} USD")

    else:
        st.warning("No data available for the selected date. Please choose another date.")


    # Create the Candlestick Chart
    fig = go.Figure(data=[
        go.Candlestick(
            x=df["Date"],
            open=df["USDI_Open"],
            high=df["USDI_High"],
            low=df["USDI_Low"],
            close=df["USDI_Price"],
            increasing_line_color='green',
            decreasing_line_color='red'
        )
    ])

    fig.update_layout(
        title="Candlestick Chart Example",
        xaxis_title="Date",
        yaxis_title="Price",
        xaxis_rangeslider_visible=False
    )
    st.markdown("<br><br>", unsafe_allow_html=True)  # Adds vertical spacing
    # Display in Streamlit
    st.header("Candlestick Chart")
    st.plotly_chart(fig)



elif options == "Classification":
    # Define the app
    st.title("Binary Classification with QDA")

    image = Image.open(class_header)
    st.image(image, use_container_width=True)


    with st.expander('Data'):
        st.write('**Raw Data**')
        #df2
        st.write(df2.head())
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
        st.write(f"Chosen Model: {name}")
        proba = chosen_model.predict_proba(input_data)

        proba_0 = proba[0][0]  # Probability of class 0 (Negative Trend)
        proba_1 = proba[0][1]  # Probability of class 1 (Positive Trend)
    
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Negative Trend (0)")
            st.progress(proba_0)  # Negative trend probability as a progress bar
            st.markdown(f"**{proba_0 * 100:.2f}%**")

        with col2:
            st.markdown("#### Positive Trend (1)")
            st.progress(proba_1)  # Positive trend probability as a progress bar
            st.markdown(f"**{proba_1 * 100:.2f}%**")
        


        if proba_1 > proba_0:
            st.success(f"The prediction is: Positive Trend")
        else:
            st.error(f"The prediction is: Negative Trend")
        # if prediction[0] == 1:
        #     st.success("The prediction is: Positive Trend")
        # else:
        #     st.error("The prediction is: Negative Trend")




elif options == "Trends":
    st.title("Trend Variables Analysis")
    # Let the user select two variables to compare
    variable1 = st.selectbox("Select the first trend variable:", df2.columns)
    variable2 = st.selectbox("Select the second trend variable:",[col for col in df2.columns if col != variable1])

    pairwise_counts = pd.crosstab(df2[variable1], df2[variable2])
    fig = px.bar(pairwise_counts, 
                    barmode='stack', 
                    labels={variable1: variable1, variable2: variable2},
                    title=f"Stacked Bar Plot of {variable1} vs {variable2}")

    # Display the plot
    st.plotly_chart(fig)
    pairwise_counts = pd.crosstab(df2[variable1], df2[variable2])
    # Prepare the data for the line plot
    data_for_line_plot = pairwise_counts.stack().reset_index(name="Count")
    
    data_for_line_plot['Combination'] = data_for_line_plot.apply(
        lambda row: f"({variable1}={row[variable1]}, {variable2}={row[variable2]})", axis=1)
    
    data_for_line_plot['HoverText'] = data_for_line_plot.apply(
        lambda row: f"{row['Combination']}, Count={row['Count']}", axis=1)
    data_for_line_plot = data_for_line_plot.sort_values(by="Combination")
    
    fig_line = go.Figure()


    fig_line.add_trace(go.Scatter(x=data_for_line_plot.index, 
                                y=data_for_line_plot['Count'], 
                                mode='lines+markers', 
                                name='Combination Counts',
                                line=dict(width=2),
                                text=data_for_line_plot['HoverText'],  
                                hoverinfo='text'))  

    fig_line.update_layout(
        title=f"Line Graph of Counts for {variable1} vs {variable2}",
        xaxis_title="Binary Combination (Hover to see details)",
        yaxis_title="Count",
        xaxis=dict(showticklabels=False),
        showlegend=True
    )

    # Display the line graph
    st.plotly_chart(fig_line)


   



# 2. **Visualize USDI Price and Volume** 
elif options == "Price & Volume":
    st.title("USDI Attributes Analysis")


    # Filter 1: Date Range
    # Define filter categories and their options
    filter_options = {
        "Date Range": ["Select Date Range"],
        "Price Range": ["Select Price Range"],
        "Volume Range": ["Select Volume Range"]
    }

    # Use st.pills() for multi-selection
    selected_filters = st.pills("Select Filters to Apply", options=list(filter_options.keys()), selection_mode="multi")

    # Show selected filters
    st.markdown(f"You selected the following filters: {', '.join(selected_filters)}")

    # Initialize filters
    filtered_df = df

    start_date = value=df['Date'].min()
    end_date = value=df['Date'].max()
    min_price = float(df['USDI_Price'].min())
    max_price = float(df['USDI_Price'].max())
    min_volume = int(df['USDI_Volume'].min())
    max_volume =int(df['USDI_Volume'].max())

    # Apply Date Range filter if selected
    if "Date Range" in selected_filters:
        start_date = st.date_input("Select Start Date", value=df['Date'].min())
        end_date = st.date_input("Select End Date", value=df['Date'].max())
        filtered_df = filtered_df[(filtered_df['Date'] >= pd.to_datetime(start_date)) & (filtered_df['Date'] <= pd.to_datetime(end_date))]

    # Apply Price Range filter if selected
    if "Price Range" in selected_filters:
        min_price = st.slider("Select Minimum Price", min_value=float(df['USDI_Price'].min()), 
                            max_value=float(df['USDI_Price'].max()), 
                            value=float(df['USDI_Price'].min()))
        max_price = st.slider("Select Maximum Price", min_value=float(df['USDI_Price'].min()), 
                            max_value=float(df['USDI_Price'].max()), 
                            value=float(df['USDI_Price'].max()))
        filtered_df = filtered_df[(filtered_df['USDI_Price'] >= min_price) & (filtered_df['USDI_Price'] <= max_price)]

    # Apply Volume Range filter if selected
    if "Volume Range" in selected_filters:
        min_volume = st.slider("Select Minimum Volume", min_value=int(df['USDI_Volume'].min()), 
                            max_value=int(df['USDI_Volume'].max()), 
                            value=int(df['USDI_Volume'].min()))
        max_volume = st.slider("Select Maximum Volume", min_value=int(df['USDI_Volume'].min()), 
                            max_value=int(df['USDI_Volume'].max()), 
                            value=int(df['USDI_Volume'].max()))
        filtered_df = filtered_df[(filtered_df['USDI_Volume'] >= min_volume) & (filtered_df['USDI_Volume'] <= max_volume)]

    date_label = f"Date: {start_date} - {end_date}"
    price_label = f"Price: ${min_price} - ${max_price}"
    volume_label = f"Volume: {min_volume} - {max_volume}"


    price, volume, price_volume = st.tabs(["USDI Price Over Time", "USDI Volume Over Time", "Price Vs Volume"])
    with price:
        st.subheader(f"USDI Price Over Time")
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        ax1.plot(filtered_df['Date'], filtered_df['USDI_Price'], label="USDI Price", marker='o')
        ax1.set_title(f"USDI Price Over Time\n{date_label}, {price_label}, {volume_label}")
        ax1.set_xlabel("Date")
        ax1.set_ylabel("Price (USD)")
        ax1.legend()
        st.pyplot(fig1)

    with volume:
        st.subheader(f"USDI Volume Over Time")
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        ax2.plot(filtered_df['Date'], filtered_df['USDI_Volume'], label="USDI Volume", marker='o', color='orange')
        ax2.set_title(f"USDI Volume Over Time\n{date_label}, {price_label}, {volume_label}")
        ax2.set_xlabel("Date")
        ax2.set_ylabel("Volume")
        ax2.legend()
        st.pyplot(fig2)

    with price_volume:
        st.subheader(f"Price vs Volume Scatter Plot")
        fig3, ax3 = plt.subplots(figsize=(8, 6))
        ax3.scatter(filtered_df['USDI_Price'], filtered_df['USDI_Volume'], alpha=0.7, color='green')
        ax3.set_title(f"Price vs Volume\n{date_label}, {price_label}, {volume_label}")
        ax3.set_xlabel("Price (USD)")
        ax3.set_ylabel("Volume")
        st.pyplot(fig3)

elif options == "Correlation Matrix":
    st.title("Correlation Matrix")
    threshold = st.slider(
        "Select correlation threshold", 
        min_value=0.0, 
        max_value=1.0, 
        value=0.5, 
        step=0.05, 
        format="%.2f"
    )
    
    correlation_matrix = df.corr()
    
    filtered_corr = correlation_matrix[correlation_matrix.abs() > threshold]
    
    st.write(filtered_corr)
    plt.figure(figsize=(10, 8)) 
    sns.heatmap(filtered_corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5, cbar=True)
    st.pyplot(plt)

