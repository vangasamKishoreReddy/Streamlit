import streamlit as st
import pandas as pd
import numpy as np

# Set page configuration
st.set_page_config(page_title="Multipages App", page_icon="👋")

# Sidebar to select pages
page = st.sidebar.selectbox("Select a page above.", ["Main Page", "Data Exploration", "About"])

# Main page content
if page == "Main Page":
    st.title("Welcome to the Main Page")
    st.write("This is the main page content.")

# Data Exploration page content
elif page == "Data Exploration":
    st.title("Data Exploration Page")

    # Load sample dataset
    @st.cache
    def load_data():
        return pd.DataFrame({
            'x': np.random.randn(100),
            'y': np.random.randn(100)
        })

    data = load_data()

    # Display dataset
    st.subheader("Sample Dataset:")
    st.dataframe(data.head())

    # Data exploration options
    if st.checkbox("Show Summary Statistics"):
        st.subheader("Summary Statistics:")
        st.write(data.describe())

    if st.checkbox("Show Scatter Plot"):
        st.subheader("Scatter Plot:")
        st.scatter_chart(data)

# About page content
elif page == "About":
    st.title("About Page")
    st.write("This is the about page content. Here you can provide information about the app, its purpose, and the team behind it.")
    st.write("Feel free to add any additional details or contact information.")

