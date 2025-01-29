import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Fetch stock data for Adani Green Energy (AGEL)
agel = yf.download("ADANIGREEN.NS", period="5y", interval="1d")

# Display first few rows
agel.head()

plt.figure(figsize=(12,6))
plt.plot(agel["Close"], label="Closing Price", color="green")
plt.title("Adani Green Energy (AGEL) Stock Price Trend")
plt.xlabel("Date")
plt.ylabel("Stock Price (INR)")
plt.legend()
plt.show()

# Prepare data for Prophet
df = agel.reset_index()[['Date', 'Close']]
df.columns = ['ds', 'y']

# Initialize and fit the model
model = Prophet()
model.fit(df)

# Create future dates
future = model.make_future_dataframe(periods=180)  # Predict next 6 months
forecast = model.predict(future)

# Plot forecast
fig = model.plot(forecast)
plt.title("Stock Price Prediction for Adani Green Energy (Next 6 Months)")
plt.show()


actual = df['y'].values[-180:]  # Last 180 days actual values
predicted = forecast['yhat'].values[-180:]

mae = mean_absolute_error(actual, predicted)
rmse = np.sqrt(mean_squared_error(actual, predicted))

print(f"Mean Absolute Error (MAE): {mae}")
print(f"Root Mean Square Error (RMSE): {rmse}")


# Calculate Moving Averages
agel["SMA_50"] = agel["Close"].rolling(window=50).mean()
agel["SMA_200"] = agel["Close"].rolling(window=200).mean()

# Define Buy/Sell conditions
agel["Buy_Signal"] = (agel["SMA_50"] > agel["SMA_200"]) & (agel["SMA_50"].shift(1) <= agel["SMA_200"].shift(1))
agel["Sell_Signal"] = (agel["SMA_50"] < agel["SMA_200"]) & (agel["SMA_50"].shift(1) >= agel["SMA_200"].shift(1))

# Plot Buy/Sell signals
plt.figure(figsize=(12,6))
plt.plot(agel["Close"], label="Closing Price", color="blue")
plt.plot(agel["SMA_50"], label="50-day SMA", color="orange")
plt.plot(agel["SMA_200"], label="200-day SMA", color="red")

# Mark Buy & Sell points
plt.scatter(agel.index[agel["Buy_Signal"]], agel["Close"][agel["Buy_Signal"]], label="Buy Signal", marker="^", color="green")
plt.scatter(agel.index[agel["Sell_Signal"]], agel["Close"][agel["Sell_Signal"]], label="Sell Signal", marker="v", color="red")

plt.title("Buy/Sell Signals for Adani Green Energy")
plt.xlabel("Date")
plt.ylabel("Stock Price (INR)")
plt.legend()
plt.show()


# Fetch data for NIFTY Green Energy Index (if available)
nifty_green = yf.download("ADANIENT.NS", period="5y", interval="1d")

# Plot NIFTY Green Energy index trend
plt.figure(figsize=(12,6))
plt.plot(nifty_green["Close"], label="NIFTY Green Energy Index", color="purple")
plt.title("NIFTY Green Energy Index Trend")
plt.xlabel("Date")
plt.ylabel("Index Value")
plt.legend()
plt.show()


# Example stock contribution data (Adjust with actual values)
stock_contributions = {
    "ADANIGREEN": 25,  # Example %
    "TATAPOWER": 20,
    "NTPC": 15,
    "SUZLON": 10,
    "JSWENERGY": 18,
    "RENEW": 12
}

# Convert to DataFrame
heatmap_data = pd.DataFrame.from_dict(stock_contributions, orient="index", columns=["Contribution"])

# Plot heatmap
plt.figure(figsize=(8,5))
sns.heatmap(heatmap_data, annot=True, cmap="Greens", linewidths=0.5)
plt.title("Stock Contribution to NIFTY Green Energy Index")
plt.show()



# Fetch financials using yfinance
stock = yf.Ticker("ADANIGREEN.NS")

# Extract key metrics
eps = stock.info.get("trailingEps", "N/A")
pe_ratio = stock.info.get("trailingPE", "N/A")
ipo_price = stock.info.get("open", "N/A")  # IPO price might not be available

print(f"Earnings Per Share (EPS): {eps}")
print(f"Price-to-Earnings (PE) Ratio: {pe_ratio}")
print(f"IPO Price: {ipo_price}")

!pip install streamlit

import streamlit as st

# Dashboard Title
st.title("Adani Green Energy Financial Dashboard")

# Stock Price Chart
st.subheader("Stock Price Trend")
st.line_chart(agel["Close"])

# Stock Prediction
st.subheader("Stock Price Prediction for Next 6 Months")
st.line_chart(forecast.set_index("ds")["yhat"])

# Buy/Sell Strategy
st.subheader("Buy/Sell Strategy Based on Moving Averages")
st.pyplot()

# Sectoral Index
st.subheader("NIFTY Green Energy Index Trend")
st.line_chart(nifty_green["Close"])

# Heatmap
st.subheader("Stock Contribution to NIFTY Green Energy Index")
st.pyplot()

# Financial KPIs
st.subheader("Key Financial Indicators")
st.write(f"**Earnings Per Share (EPS):** {eps}")
st.write(f"**Price-to-Earnings (PE) Ratio:** {pe_ratio}")
st.write(f"**IPO Price:** {ipo_price}")



!pip install streamlit
!pip install pyngrok


import streamlit as st

with open("app.py", "w") as f:
  f.write("import streamlit as stst.title")
  st.write("This is a Streamlit app running on Google Colab.")


from pyngrok import ngrok

# Start Streamlit in the background
!streamlit run app.py &

# Create a public URL using ngrok
url = ngrok.connect(8501).public_url
print(f"Streamlit app is running at: {url}")
