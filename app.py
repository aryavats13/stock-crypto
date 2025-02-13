import streamlit as st
from stock_page import show_stock_page  # Import the function
from home_page import home_page
# Initialize session state
with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)


if 'page' not in st.session_state:
    st.session_state.page = 'home'
if 'stock' not in st.session_state:
    st.session_state.stock = 'AAPL'  # Default stock

# Main app logic
if st.session_state.page == 'home':
    home_page()
elif st.session_state.page == 'analysis':
    show_stock_page()