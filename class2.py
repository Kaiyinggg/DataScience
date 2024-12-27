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


final = 'FINAL_USO.csv'
gold = 'gold.csv'
gold_header = 'Gold_Header.jpeg'
ada = 'adaboost_model.sav'
dnn = 'dnn_model.sav'

df_gold = pd.read_csv(gold)
df_final = pd.read_csv(final)
df_final['Date'] = pd.to_datetime(df_final['Date'])
ada_model = pickle.load(open(ada, 'rb'))
dnn_model = pickle.load(open(dnn, 'rb'))


# Sidebar: Options for Users
st.sidebar.header("Menu")
# options = st.sidebar.radio("Choose the desired view:", ["Dashboard" ,"Classification", "Trends", "Price & Volume"])
# Sidebar stylish navigation
with st.sidebar:
    # Main Navigation Menu
    options = option_menu(
        "Navigation", 
        ["Dashboard", "Trends", "Price & Volume","Statistics","Correlation Matrix", "Prediction"], 
        icons=['house', 'bar-chart', 'graph-up-arrow', 'calculator', 'diagram-3', 'search'], 
        menu_icon="list", 
        default_index=0
    )

    
if options == "Dashboard":
    # Get the latest date in the dataset
    #latest_date = df["Date"].max()
    latest_date = df["Date"].iloc[-2]

    # Add a Date Picker for user to select a date
    st.title("US Dollar Price and Volume Metrics")
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
        title="Candlestick Chart for US Dollar",
        xaxis_title="Date",
        yaxis_title="Price",
        xaxis_rangeslider_visible=False
    )
    st.markdown("<br><br>", unsafe_allow_html=True)  # Adds vertical spacing
    # Display in Streamlit
    st.plotly_chart(fig)


    # Additional Section: Gold Price and Volume Metrics
    latest_gold_date = df_final["Date"].iloc[-2]
    st.title("Gold Price and Volume Metrics")

    # Add a Date Picker for Gold
    selected_gold_date = st.date_input(
        "Select a Date for Gold",
        min_value=df_final["Date"].min(),
        max_value=df_final["Date"].max(),
        value=latest_gold_date,
        key="gold_date_picker",
    )

    # Find the row for the selected date
    selected_gold_row = df_final[df_final['Date'] == pd.to_datetime(selected_gold_date)]

    # Check if the selected date is valid
    if not selected_gold_row.empty:
        # Get the previous day row (if available)
        previous_gold_row = df_final[df_final['Date'] == pd.to_datetime(selected_gold_date) - pd.Timedelta(days=1)]

        # If there is no previous row, inform the user
        if previous_gold_row.empty:
            st.warning("No data available for the previous day.")
            previous_gold_row = selected_gold_row  # Treat as the same row for first entry

        # Extract values for the selected and previous rows
        latest_gold_price = selected_gold_row["Adj Close"].values[0]
        previous_gold_price = previous_gold_row["Adj Close"].values[0]
        gold_price_change = latest_gold_price - previous_gold_price
        gold_price_change_pct = (gold_price_change / previous_gold_price) * 100 if previous_gold_price != 0 else 0

        latest_gold_volume = selected_gold_row["Volume"].values[0]
        previous_gold_volume = previous_gold_row["Volume"].values[0]
        gold_volume_change = latest_gold_volume - previous_gold_volume
        gold_volume_change_pct = (gold_volume_change / previous_gold_volume) * 100 if previous_gold_volume != 0 else 0

        latest_gold_high = selected_gold_row["High"].values[0]
        latest_gold_low = selected_gold_row["Low"].values[0]
        gold_high_low_difference = latest_gold_high - latest_gold_low

        # Use columns to organize the metrics
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(label="Gold Price", value=f"${latest_gold_price:.2f}", delta=f"{gold_price_change:.2f} USD")

        with col2:
            st.metric(label="Gold Volume", value=f"{latest_gold_volume:,}", delta=f"{gold_volume_change:.2f} USD")

        with col3:
            st.metric(label="Gold High (Latest)", value=f"${latest_gold_high:.2f}", delta=f"{gold_high_low_difference:.2f} USD")

    else:
        st.warning("No data available for the selected date. Please choose another date.")


    # Create the Candlestick Chart
    fig = go.Figure(data=[
        go.Candlestick(
            x=df_final["Date"],
            open=df_final["Open"],
            high=df_final["High"],
            low=df_final["Low"],
            close=df_final["Adj Close"],
            increasing_line_color='green',
            decreasing_line_color='red'
        )
    ])

    fig.update_layout(
        title="Candlestick Chart for gold",
        xaxis_title="Date",
        yaxis_title="Price",
        xaxis_rangeslider_visible=False
    )
    st.markdown("<br><br>", unsafe_allow_html=True)  # Adds vertical spacing
    # Display in Streamlit
    st.plotly_chart(fig)

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

elif options == "Prediction":   
    prediction_type = option_menu(
        "Prediction Type",
        ["Classification", "Regression"],
        icons=['check-circle', 'graph-up'],
        menu_icon="list",
        default_index=0
    )

    if prediction_type == "Classification":  
        model_options = option_menu(
                "Classification Model Selection",
                ["Quadratic Discriminant Analysis (QDA) Model", "Multinomial Naive Bayes (MNB) Model"],
                icons=['calculator', 'file-earmark-text'],
                menu_icon="list",
                default_index=0
            )

        chosen_model = qda_model if model_options == "Quadratic Discriminant Analysis (QDA) Model" else mnb_model
        name = "Quadratic Discriminant Analysis" if model_options == "Quadratic Discriminant Analysis (QDA) Model" else "Multinomial Naive Bayes"
        
        # Define the app
        st.title("USDI Trend Prediction")

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
        
    elif prediction_type == "Regression":
            model_options = option_menu(
                "Regression Model Selection",
                ["AdaBoost Model", " Deep Neural Networks (DNN) Model"],
                icons=['bar-chart-line', 'layers'],
                menu_icon="list",
                default_index=0
            )

            chosen_model = ada_model if model_options == "AdaBoost Model" else dnn_model
            name = "AdaBoost Regressor" if model_options == "AdaBoost Model" else "Deep Neural Network"

            st.title("Gold Price Prediction")          
            image = Image.open(gold_header)
            st.image(image, use_container_width=True)

            with st.expander('Data'):
                st.write('**Raw Data**')
                st.write(df_gold.head())  

            # Input features
            st.header("Input Features")

            try:
                GDX_Close = st.number_input("GDX Close", min_value=0.0, step=100.0)
                GDX_High = st.number_input("GDX High", min_value=0.0, step=100.0)
                SF_Low = st.number_input("SF Low", min_value=0.0, step=100.0)
                SF_Price = st.number_input("SF Price", min_value=0.0, step=100.0)
                EG_Low = st.number_input("EG Low", min_value=0.0, step=100.0)
                EG_Open = st.number_input("EG Open", min_value=0.0, step=100.0)
                PLT_Price = st.number_input("PLT Price", min_value=0.0, step=100.0)
                PLT_High = st.number_input("PLT High", min_value=0.0, step=100.0)
            except ValueError:
                st.error("Please enter valid numeric values")

            # Create a DataFrame for prediction
            input_data = {
                'GDX_Close': GDX_Close,
                'GDX_High': GDX_High,
                'SF_Low': SF_Low,
                'SF_Price': SF_Price,
                'EG_Low': EG_Low,
                'EG_Open': EG_Open,
                'PLT_Price': PLT_Price,
                'PLT_High': PLT_High
            }

            input_df = pd.DataFrame([input_data])

            # Predict button functionality
            if st.button("Predict"):
                prediction = chosen_model.predict(input_df)
                predicted_price = prediction[0] 

                # Display the prediction result
                st.success(f"The predicted gold price is: ${predicted_price}")
                
elif options == "Statistics":
    # Extract year from the 'Date' column and add it as a new column
    df_final['Year'] = df_final['Date'].dt.year
    
    # Get the latest year from the data
    latest_gold_year = df_final['Year'].max()
    
    # Title for the app
    st.title("Commodities Summary Statistics")
    
    # Add a Year Selector (select a year)
    selected_gold_year = st.selectbox(
        "Select a Year for Commodities Analysis",
        options=df_final['Year'].unique(),
        index=list(df_final['Year'].unique()).index(latest_gold_year)
    )
    
    # Filter the DataFrame by the selected year
    selected_year_df = df_final[df_final['Year'] == selected_gold_year]
    
    # Check if there is data for the selected year
    if not selected_year_df.empty:
        # List of available columns for the user to select
        available_columns = [
            'GDX_Close', 'GDX_High', 'SF_Low', 'SF_Price', 'EG_low', 
            'EG_open', 'PLT_Price', 'PLT_High', 'Adj Close'
        ]
        
        # Allow user to select columns to display summary statistics
        selected_columns = st.multiselect(
            "Select Attributes to Analyze", 
            options=available_columns,
            default=available_columns  # Default to all columns selected
        )
        
        if selected_columns:
            # Calculate summary statistics (mean, median, max, min) for the selected columns
            summary = selected_year_df[selected_columns].describe().T[['mean', '50%', 'max', 'min']]
            summary.rename(columns={'50%': 'median'}, inplace=True)  # Rename 50% to median
    
            st.header(f"Summary Statistics for Commodities Metrics in {selected_gold_year}")
    
            # Display the statistics grouped by attribute
            for col in selected_columns:
                st.markdown(f"#### **{col}**")
                col1, col2, col3, col4 = st.columns(4)  # Four columns: Mean, Median, Max, Min
    
                with col1:
                    st.metric(label=f"{col} Mean", value=f"${summary.loc[col, 'mean']:.2f}")
                with col2:
                    st.metric(label=f"{col} Median", value=f"${summary.loc[col, 'median']:.2f}")
                with col3:
                    st.metric(label=f"{col} Max", value=f"${summary.loc[col, 'max']:.2f}")
                with col4:
                    st.metric(label=f"{col} Min", value=f"${summary.loc[col, 'min']:.2f}")
    
        else:
            st.write("Please select at least one attribute to see the summary statistics.")
    else:
        st.warning(f"No data available for the year {selected_gold_year}. Please select a different year.")
      

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
        st.subheader("Date Range")
        start_date = st.date_input("Select Start Date", value=df['Date'].min())
        end_date = st.date_input("Select End Date", value=df['Date'].max())
        filtered_df = filtered_df[(filtered_df['Date'] >= pd.to_datetime(start_date)) & (filtered_df['Date'] <= pd.to_datetime(end_date))]

    # Apply Price Range filter if selected
    if "Price Range" in selected_filters:
        st.subheader("Price Range")
        min_price = st.slider("Select Minimum Price", min_value=float(df['USDI_Price'].min()), 
                            max_value=float(df['USDI_Price'].max()), 
                            value=float(df['USDI_Price'].min()))
        max_price = st.slider("Select Maximum Price", min_value=float(df['USDI_Price'].min()), 
                            max_value=float(df['USDI_Price'].max()), 
                            value=float(df['USDI_Price'].max()))
        filtered_df = filtered_df[(filtered_df['USDI_Price'] >= min_price) & (filtered_df['USDI_Price'] <= max_price)]

    # Apply Volume Range filter if selected
    if "Volume Range" in selected_filters:
        st.subheader("Volume Range")
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
    st.title("Correlation Matrix for USDI")
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

    st.title("Correlation Matrix for Gold")

    selected_columns = ['Adj Close', 'SP_Ajclose', 'DJ_Ajclose', 'EG_Ajclose', 'EU_Price', 'OF_Price', 'OS_Price', 'SF_Price', 
                        'USB_Price', 'PLT_Price', 'PLD_Price', 'RHO_PRICE', 'USDI_Price', 'GDX_Adj Close', 'USO_Adj Close']  

    # Apply the column filter to the dataframe
    filtered_df_final = df_final[selected_columns]

    # Select the correlation threshold
    threshold = st.slider(
        "Select correlation threshold", 
        min_value=0.0, 
        max_value=1.0, 
        value=0.5, 
        step=0.05, 
        format="%.2f",
        key="correlation_threshold"
    )

    # Calculate the correlation matrix for the selected columns
    correlation_matrix = filtered_df_final.corr()

    # Filter the correlation matrix based on the threshold
    filtered_df_corr = correlation_matrix[correlation_matrix.abs() > threshold]

    # Display the filtered correlation matrix
    st.write(filtered_df_corr)

    # Plot the heatmap
    plt.figure(figsize=(10, 8)) 
    sns.heatmap(filtered_df_corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5, cbar=True)
    st.pyplot(plt)
