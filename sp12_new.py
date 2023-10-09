#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import pandas as pd
import pickle

# Define the preprocessing function
def preprocess_data(data):
    # Changing Datatype of 'Transaction Date' feature available to proper date format
    data['Transaction Date'] = pd.to_datetime(data['Transaction Date'], errors='coerce', format="%d-%m-%Y")

    # Extracting date features
    data['Year'] = data['Transaction Date'].dt.year
    data['Month'] = data['Transaction Date'].dt.month
    data['Day'] = data['Transaction Date'].dt.day
    data['DayOfWeek'] = data['Transaction Date'].dt.dayofweek

    # Encoding categorical features
    features_to_encode = ['Country', 'Sales Type', 'Brand', 'Area', 'Product Line', 'Product Category']

    # Create new columns based on user input and drop the original columns
    for feature in features_to_encode:
        new_column_name = f'{feature}_{data[feature][0]}'  # New column name based on user input
        data[new_column_name] = 1  # Set the value to 1 for the new column
        data.drop([feature], axis=1, inplace=True)  # Drop the original column

    # Creating lag and rolling mean features
    data['Sold Units Lag 1'] = data['Sold Units'].shift(1)
    data['Sold Units Lag 30'] = data['Sold Units'].shift(30)
    data['Sold Units Rolling Mean 3'] = data['Sold Units'].rolling(window=3, min_periods=1).mean()

    data = data.fillna(0)
    data = data.drop('Transaction Date', axis=1)

    return data

# Load the trained models
with open('sales_xgb_model.pkl', 'rb') as file:
    regressor = pickle.load(file)  
    
st.title("Sales Forecasting System for Retail Industry")

# Get input from user
country = st.text_input("Country")
sales_type = st.text_input("Sales Type")
transaction_date = st.text_input("Transaction Date (DD-MM-YYYY)")
brand = st.text_input("Brand")
area = st.text_input("Area")
product_line = st.text_input("Product Line")
product_category = st.text_input("Product Category")
sold_units = st.number_input("Sold Units")
unit_cost_price = st.number_input("Unit Cost Price", step=0.01)

# Create a DataFrame from the user input
user_data = {
    'Country': [country],
    'Sales Type': [sales_type],
    'Transaction Date': [transaction_date],
    'Brand': [brand],
    'Area': [area],
    'Product Line': [product_line],
    'Product Category': [product_category],
    'Sold Units': [sold_units],
    'Net_Cost_Price': [unit_cost_price],
}

user_df = pd.DataFrame(user_data)

# Add a "Predict" button
if st.button("Predict"):
    # Perform preprocessing
    processed_df = preprocess_data(user_df)

    original_dataset = pd.read_csv('retail_sales_preprocessed.csv')
    original_dataset = original_dataset.drop(['Unnamed: 0', 'Transaction Date', 'Net_SalesValue'], axis=1)

    for col in original_dataset.columns:
        if col not in processed_df.columns:
            processed_df[col] = 0

    # Reorder the columns to match the order of the original dataset
    processed_df = processed_df[original_dataset.columns]

    # Modeling
    st.header("Model Predictions")

    # Predict using Decision Tree model
    st.subheader("Decision Tree Model Predictions")
    y_pred_dt = dt_regressor.predict(processed_df)
    st.write("Prediction done by Decision Tree model is:", y_pred_dt)

    # Predict using Random Forest model
    st.subheader("Random Forest Model Predictions")
    y_pred_rf = rf_regressor.predict(processed_df)
    st.write("Prediction done by Random Forest model is:", y_pred_rf)

    # Create a DataFrame with original inputs and predictions
    output_df = pd.concat([user_df.reset_index(drop=True),
                           pd.DataFrame({'Prediction (Decision Tree)': y_pred_dt,
                                         'Prediction (Random Forest)': y_pred_rf})], axis=1)

    # Provide a link to download the DataFrame as CSV
    st.subheader("Download Predictions")
    st.write("Download the predictions as a CSV file.")
    st.dataframe(output_df, height=300)
    csv = output_df.to_csv(index=False)
    st.download_button(label="Download CSV", data=csv, file_name="predictions.csv", mime="text/csv")


# In[ ]:




