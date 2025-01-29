pip install streamlit pandas numpy matplotlib seaborn plotly scikit-learn yfinance google-auth google-auth-oauthlib google-auth-httplib2 google-auth google-api-python-client
pip install nsepy
import streamlit as st
import pandas as pd
import time
from nsepy import get_history
from datetime import date

# Define function to fetch data with delay
def fetch_nse_data(symbol, start_date, end_date):
    time.sleep(5)  # Adding delay to avoid NSE rate limits
    return get_history(symbol=symbol, index=True, start=start_date, end=end_date)

# Fetch Nifty Green Energy index data
try:
    nifty_green = fetch_nse_data("NIFTY GREEN ENERGY", date(2019,1,1), date(2024,1,1))
    st.write(nifty_green.head())
except Exception as e:
    st.error(f"Error fetching data: {e}")


if not nifty_green.empty:
    st.subheader("Nifty Green Energy Index Data (Last 5 Entries)")
    st.dataframe(nifty_green.tail())  # Display last 5 rows
else:
    st.warning("No data found. Try again later.")


import plotly.express as px

if not nifty_green.empty:
    fig = px.line(nifty_green, x=nifty_green.index, y="Close", title="Nifty Green Energy Live Chart")
    st.plotly_chart(fig)


# Add Moving Averages
nifty_green['SMA50'] = nifty_green['Close'].rolling(window=50).mean()
nifty_green['SMA200'] = nifty_green['Close'].rolling(window=200).mean()

# Define Buy/Sell Signals
def buy_sell_signals(data):
    buy = []
    sell = []
    
    for i in range(len(data)):
        if data["SMA50"][i] > data["SMA200"][i]:  # Buy Signal
            buy.append(data["Close"][i])
            sell.append(None)
        elif data["SMA50"][i] < data["SMA200"][i]:  # Sell Signal
            sell.append(data["Close"][i])
            buy.append(None)
        else:
            buy.append(None)
            sell.append(None)
    
    return buy, sell

nifty_green["Buy_Signal"], nifty_green["Sell_Signal"] = buy_sell_signals(nifty_green)




import matplotlib.pyplot as plt

if not nifty_green.empty:
    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(nifty_green["Close"], label="Close Price", alpha=0.5)
    ax.plot(nifty_green["SMA50"], label="50-Day SMA", linestyle="dashed")
    ax.plot(nifty_green["SMA200"], label="200-Day SMA", linestyle="dashed")

    # Plot Buy Signals
    ax.scatter(nifty_green.index, nifty_green["Buy_Signal"], label="BUY", marker="^", color="green", alpha=1)
    
    # Plot Sell Signals
    ax.scatter(nifty_green.index, nifty_green["Sell_Signal"], label="SELL", marker="v", color="red", alpha=1)
    
    ax.set_title("Buy/Sell Predictions using SMA Strategy")
    ax.legend()
    st.pyplot(fig)



sector_data = {
    "Stock": ["Stock A", "Stock B", "Stock C", "Stock D"],
    "Contribution (%)": [30, 25, 20, 25]
}
sector_df = pd.DataFrame(sector_data)
st.dataframe(sector_df)



import seaborn as sns

fig, ax = plt.subplots(figsize=(8,4))
sns.heatmap(sector_df.set_index("Stock").T, cmap="coolwarm", annot=True, linewidths=0.5)
st.pyplot(fig)


financial_metrics = {
    "Metric": ["Earnings Per Share (EPS)", "Price-to-Earnings (PE) Ratio", "IPO Price"],
    "Value": [12.5, 18.7, 250]  # Replace with actual data
}

financial_df = pd.DataFrame(financial_metrics)
st.subheader("Sector Financial Metrics")
st.dataframe(financial_df)
