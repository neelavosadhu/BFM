import streamlit as st
import pandas as pd
import numpy as np

# Set the title of the Streamlit app
st.title("Simple Graph using Streamlit")

# Generate random data
data = pd.DataFrame(
    np.random.randn(20, 2),
    columns=['X', 'Y']
)

# Display the line chart
st.line_chart(data)
