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

