import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime, timedelta
import pickle

# Function to load stock data
def load_stock_data(ticker):
    stock_data = yf.download(ticker, start="2020-01-01", end="2025-01-26")
    return stock_data

# Function to load stored predictions
def load_predictions(ticker):
    try:
        with open(f"{ticker}_predictions.pkl", "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        return None

# Load actual stock data and predictions
st.title("Stock Price and Sales Prediction Dashboard")

company = st.selectbox("Select Company", ["AAPL", "GOOGL", "MSFT", "AMZN"])
stock_data = load_stock_data(company)
predictions = load_predictions(company)

# Display actual and predicted prices with error percentage
if predictions:
    actual_price = stock_data.iloc[-1]['Close']
    predicted_price = predictions['predicted_price']
    error_percentage = abs((predicted_price - actual_price) / actual_price) * 100
    
    st.markdown(f"### Actual Price: **${actual_price:.2f}**")
    st.markdown(f"### Predicted Price: **${predicted_price:.2f}**")
    st.markdown(f"### Error: **{error_percentage:.2f}%**")

# Time range filter
st.subheader("Stock Price & Sales Prediction")
time_filter = st.selectbox("Select Time Range", ["1 Day", "1 Week", "1 Month", "3 Months", "6 Months", "1 Year", "3 Years", "5 Years", "All"])

def filter_data(data, time_filter):
    end_date = data.index[-1]
    if time_filter == "1 Day":
        start_date = end_date - timedelta(days=1)
    elif time_filter == "1 Week":
        start_date = end_date - timedelta(weeks=1)
    elif time_filter == "1 Month":
        start_date = end_date - timedelta(weeks=4)
    elif time_filter == "3 Months":
        start_date = end_date - timedelta(weeks=12)
    elif time_filter == "6 Months":
        start_date = end_date - timedelta(weeks=24)
    elif time_filter == "1 Year":
        start_date = end_date - timedelta(weeks=52)
    elif time_filter == "3 Years":
        start_date = end_date - timedelta(weeks=156)
    elif time_filter == "5 Years":
        start_date = end_date - timedelta(weeks=260)
    else:
        start_date = data.index[0]
    return data.loc[start_date:end_date]

filtered_stock_data = filter_data(stock_data, time_filter)

# Plot the graph
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(filtered_stock_data.index, filtered_stock_data['Close'], label='Actual Price', color='blue')
if predictions:
    ax.plot(filtered_stock_data.index, predictions['predicted_values'], label='Predicted Price', linestyle='dashed', color='red')
ax.set_title(f"Stock Price and Prediction ({company})")
ax.legend()
st.pyplot(fig)

# Include News Section
st.subheader("Latest News for " + company)
news_data = ["News 1: Stock surges on earnings report", "News 2: Market sees volatility", "News 3: CEO announces new strategy"]
for news in news_data:
    st.write(news)
