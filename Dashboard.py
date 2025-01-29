import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import time

# Function to fetch live stock data
def fetch_live_stock(ticker="ADANIGREEN.NS"):
    stock = yf.Ticker(ticker)
    data = stock.history(period="1d", interval="1m")  # 1-minute interval
    return data

# Streamlit App
st.set_page_config(page_title="Live Stock Dashboard", layout="wide")

st.title("📈 Live Stock Dashboard - Adani Green Energy")

# Sidebar for user input
ticker = st.sidebar.text_input("Enter Stock Ticker:", "ADANIGREEN.NS")
update_interval = st.sidebar.slider("Update Interval (seconds)", 5, 60, 10)

# Create a placeholder for live stock price
live_price_placeholder = st.empty()
chart_placeholder = st.empty()

# Fetch initial data
prices = []

while True:
    # Fetch latest data
    data = fetch_live_stock(ticker)
    latest_price = data['Close'].iloc[-1]
    prices.append(latest_price)

    # Update Live Price
    live_price_placeholder.metric(label="Current Price (₹)", value=f"{latest_price:.2f}")

    # Update Live Chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=prices, mode='lines+markers', name='Live Price'))
    fig.update_layout(title=f"Live Stock Price of {ticker}",
                      xaxis_title="Time",
                      yaxis_title="Price (₹)",
                      template="plotly_dark")
    
    chart_placeholder.plotly_chart(fig, use_container_width=True)

    time.sleep(update_interval)
