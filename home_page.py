import streamlit as st

def home_page():
    st.markdown('<h1 class="main-title">Stock Market Predictor</h1>', unsafe_allow_html=True)
    
    st.markdown('''
    <div class="intro-box">
        <p>Welcome to the Stock Market Predictor. This tool helps you analyze and predict stock trends.</p>
        <p>Choose a stock from the list below to begin your analysis.</p>
    </div>
    ''', unsafe_allow_html=True)

    # List of stocks (you can expand this list)
    stocks = [
        "AAPL", "GOOGL", "MSFT", "AMZN", "META", "TSLA", "NVDA", "JPM", "JNJ", "V",
        "PG", "UNH", "MA", "HD", "DIS", "BAC", "ADBE", "CRM", "NFLX", "CMCSA"
    ]

    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        selected_stock = st.selectbox("Choose a stock", stocks, key="stock_selector")
        st.markdown('<div class="spacer"></div>', unsafe_allow_html=True)
        if st.button("Analyze Stock", key="analyze_button"):
            st.session_state.stock = selected_stock
            st.session_state.page = 'analysis'
            st.experimental_rerun()

    st.markdown('''
    <div class="footer">
        <p>© 2024 Stock Market Predictor. All rights reserved.</p>
    </div>
    ''', unsafe_allow_html=True)