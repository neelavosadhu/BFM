import streamlit as st
import pandas as pd
import time
from nsepy import get_history
from datetime import date
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------- Step 1: Fetch Nifty Green Energy Data --------------------------

@st.cache_data
def fetch_nse_data(symbol, start_date, end_date):
    """Fetches historical index data from NSE with a delay to prevent rate limits."""
    time.sleep(5)
    return get_history(symbol=symbol, index=True, start=start_date, end=end_date)

# Fetch Data
st.title("📊 Nifty Green Energy Dashboard")
st.sidebar.header("Settings")

start_date = date(2019, 1, 1)
end_date = date(2024, 1, 1)

try:
    nifty_green = fetch_nse_data("NIFTY GREEN ENERGY", start_date, end_date)
    if not nifty_green.empty:
        st.subheader("📈 Nifty Green Energy Index Data")
        st.dataframe(nifty_green.tail())
    else:
        st.warning("⚠ No data found. Try again later.")
except Exception as e:
    st.error(f"❌ Error fetching data: {e}")

# -------------------------- Step 2: Live Chart for Sectoral Index --------------------------

if not nifty_green.empty:
    st.subheader("📉 Live Nifty Green Energy Chart")
    fig = px.line(nifty_green, x=nifty_green.index, y="Close", title="Nifty Green Energy Index")
    st.plotly_chart(fig)

# -------------------------- Step 3: Stock Prediction (Buy/Sell Decision) --------------------------

def buy_sell_signals(data):
    """Generates Buy/Sell signals using 50-day and 200-day moving averages."""
    data['SMA50'] = data['Close'].rolling(window=50).mean()
    data['SMA200'] = data['Close'].rolling(window=200).mean()
    
    buy = []
    sell = []
    
    for i in range(len(data)):
        if data["SMA50"][i] > data["SMA200"][i]:  
            buy.append(data["Close"][i])
            sell.append(None)
        elif data["SMA50"][i] < data["SMA200"][i]:  
            sell.append(data["Close"][i])
            buy.append(None)
        else:
            buy.append(None)
            sell.append(None)
    
    data["Buy_Signal"] = buy
    data["Sell_Signal"] = sell
    return data

if not nifty_green.empty:
    nifty_green = buy_sell_signals(nifty_green)
    st.subheader("💹 Buy/Sell Predictions")
    
    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(nifty_green["Close"], label="Close Price", alpha=0.5)
    ax.plot(nifty_green["SMA50"], label="50-Day SMA", linestyle="dashed")
    ax.plot(nifty_green["SMA200"], label="200-Day SMA", linestyle="dashed")
    ax.scatter(nifty_green.index, nifty_green["Buy_Signal"], label="BUY", marker="^", color="green", alpha=1)
    ax.scatter(nifty_green.index, nifty_green["Sell_Signal"], label="SELL", marker="v", color="red", alpha=1)
    
    ax.set_title("Buy/Sell Predictions using SMA Strategy")
    ax.legend()
    st.pyplot(fig)

# -------------------------- Step 4: Stock Contribution Heatmap --------------------------

st.subheader("🔥 Stock Contribution to Nifty Green Energy Index")

sector_data = {
    "Stock": ["Stock A", "Stock B", "Stock C", "Stock D"],
    "Contribution (%)": [30, 25, 20, 25]
}

sector_df = pd.DataFrame(sector_data)
st.dataframe(sector_df)

fig, ax = plt.subplots(figsize=(8,4))
sns.heatmap(sector_df.set_index("Stock").T, cmap="coolwarm", annot=True, linewidths=0.5)
st.pyplot(fig)

# -------------------------- Step 5: Financial Metrics --------------------------

st.subheader("📊 Sector Financial Metrics")

financial_metrics = {
    "Metric": ["Earnings Per Share (EPS)", "Price-to-Earnings (PE) Ratio", "IPO Price"],
    "Value": [12.5, 18.7, 250]  # Replace with actual data
}

financial_df = pd.DataFrame(financial_metrics)
st.dataframe(financial_df)

st.success("✅ Dashboard is ready! Deploy this on Streamlit Cloud.")

