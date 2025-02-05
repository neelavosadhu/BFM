import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.dates as mdates
import yfinance as yf

# Define the stock ticker and fetch data from Yahoo Finance
ticker = "AAPL"  # Example ticker, replace with your desired ticker
stock = yf.Ticker(ticker)
data = stock.history(period="5y")  # Fetching last 5 years of data

# Data Preprocessing
data.reset_index(inplace=True)
data['Date'] = pd.to_datetime(data['Date'])
data['Month'] = data['Date'].dt.month
data['Year'] = data['Date'].dt.year

# Feature selection and target variable
X = data[['Month', 'Year']]
y = data['Close']  # Assuming 'Close' price is used for sales prediction

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict sales
data['Predicted_Sales'] = model.predict(X)

# Plotting the results
fig, ax = plt.subplots()

ax.plot(data['Date'], data['Close'], label='Actual Sales', color='blue')
ax.plot(data['Date'], data['Predicted_Sales'], label='Predicted Sales', color='orange')

# Adding buy and sell markers with arrows
data['Previous_Close'] = data['Close'].shift(1)  # Shift the entire column
for i, row in data.iterrows():
    if row['Close'] > row['Previous_Close']:  # Example condition for Buy
        ax.annotate('↑', (row['Date'], row['Close']), color='green', textcoords="offset points", xytext=(0,10), ha='center')
    elif row['Close'] < row['Previous_Close']:  # Example condition for Sell
        ax.annotate('↓', (row['Date'], row['Close']), color='red', textcoords="offset points", xytext=(0,-10), ha='center')

# Adding filters
ax.set_title('Sales Prediction with Buy/Sell Data')
ax.set_xlabel('Date')
ax.set_ylabel('Sales')
ax.legend()

# Create a function to filter data based on the selected range
def filter_data(range):
    if range == '1 month':
        return data[data['Date'] >= data['Date'].max() - pd.DateOffset(months=1)]
    elif range == '3 months':
        return data[data['Date'] >= data['Date'].max() - pd.DateOffset(months=3)]
    elif range == '6 months':
        return data[data['Date'] >= data['Date'].max() - pd.DateOffset(months=6)]
    elif range == '1 year':
        return data[data['Date'] >= data['Date'].max() - pd.DateOffset(years=1)]
    elif range == '3 years':
        return data[data['Date'] >= data['Date'].max() - pd.DateOffset(years=3)]
    elif range == '5 years':
        return data[data['Date'] >= data['Date'].max() - pd.DateOffset(years=5)]
    else:
        return data

# Example usage of the filter function
filtered_data = filter_data('1 year')
plt.show()
