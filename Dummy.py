
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.dates as mdates
import gspread
from oauth2client.service_account import ServiceAccountCredentials

# Google Sheets authentication
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
creds = ServiceAccountCredentials.from_json_keyfile_name('path_to_your_service_account.json', scope)
client = gspread.authorize(creds)

# Open the Google Sheet
sheet = client.open("Your Google Sheet Name").sheet1

# Fetch all records
records = sheet.get_all_records()
data = pd.DataFrame.from_records(records)

# Data Preprocessing
data['Date'] = pd.to_datetime(data['Date'])
data['Month'] = data['Date'].dt.month
data['Year'] = data['Date'].dt.year

# Feature selection and target variable
X = data[['Month', 'Year']]
y = data['Sales']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict sales
data['Predicted_Sales'] = model.predict(X)

# Plotting the results
fig, ax = plt.subplots()

ax.plot(data['Date'], data['Sales'], label='Actual Sales', color='blue')
ax.plot(data['Date'], data['Predicted_Sales'], label='Predicted Sales', color='orange')

# Adding buy and sell markers with arrows
for i, row in data.iterrows():
    if row['Buy'] > 0:
        ax.annotate('↑', (row['Date'], row['Sales']), color='green', textcoords="offset points", xytext=(0,10), ha='center')
    if row['Sell'] > 0:
        ax.annotate('↓', (row['Date'], row['Sales']), color='red', textcoords="offset points", xytext=(0,-10), ha='center')

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
