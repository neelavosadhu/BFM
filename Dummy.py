import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.express as px

# Dummy Mutual Fund Data
fund_data = {
    "fund_name": "Groww Value Fund Direct Growth",
    "type": "Equity",
    "risk_level": "Very High Risk",
    "return_3y": "13.83%",
    "change_1d": "+1.26%",
    "nav": 29.08,
    "min_sip": 500,
    "fund_size": 46.04
}

# Fetch Stock Data (Example: NIFTY 50)
ticker = "^NSEI"  # Nifty 50 Index
stock = yf.Ticker(ticker)
history = stock.history(period="3y")
history["Date"] = history.index

# Sidebar - SIP Investment Section
st.sidebar.header("SIP Investment")
sip_amount = st.sidebar.number_input("SIP Amount (₹)", min_value=500, step=100)
sip_date = st.sidebar.selectbox("SIP Date", list(range(1, 29)))
st.sidebar.button("Start SIP")

# Header
st.title(fund_data["fund_name"])
st.write(f"**{fund_data['type']} | {fund_data['risk_level']}**")
st.metric("3Y Annualized Return", fund_data["return_3y"], fund_data["change_1d"])

# NAV Line Chart
fig = px.line(history, x="Date", y="Close", title="NAV Over Time", labels={"Close": "NAV (₹)"})
st.plotly_chart(fig, use_container_width=True)

# Holding Analysis
st.subheader("Holding Analysis")
holding_data = pd.DataFrame({
    "Category": ["Equity", "Debt", "Cash"],
    "Percentage": [91.4, 5.7, 2.8]
})
holding_fig = px.pie(holding_data, names="Category", values="Percentage", title="Equity/Debt/Cash Split")
st.plotly_chart(holding_fig, use_container_width=True)

# Sector Allocation
st.subheader("Equity Sector Allocation")
sector_data = pd.DataFrame({
    "Sector": ["Financial", "Others", "Energy", "Automobile", "Technology", "Construction", "Services", "Healthcare"],
    "Percentage": [38.1, 13.8, 9.0, 9.0, 8.8, 8.5, 7.3, 5.6]
})
sector_fig = px.pie(sector_data, names="Sector", values="Percentage", title="Sector Allocation")
st.plotly_chart(sector_fig, use_container_width=True)

# Fund Details
st.subheader("Fund Details")
st.write(f"**NAV:** ₹{fund_data['nav']} (as of latest available date)")
st.write(f"**Minimum SIP Amount:** ₹{fund_data['min_sip']}")
st.write(f"**Fund Size:** ₹{fund_data['fund_size']} Cr")

