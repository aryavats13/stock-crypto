import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from prophet import Prophet
import plotly.graph_objects as go
from datetime import datetime, timedelta

def fetch_stock_data(symbol):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=730)
    df = yf.download(symbol, start=start_date, end=end_date)
    return df

def calculate_technical_indicators(df):
    df['RSI'] = calculate_rsi(df['Close'])
    df['MACD'], df['Signal'] = calculate_macd(df['Close'])
    return df

def calculate_rsi(prices, period=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_macd(prices, slow=26, fast=12, signal=9):
    exp1 = prices.ewm(span=fast, adjust=False).mean()
    exp2 = prices.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line

def plot_technical_indicators(df):
    fig_rsi = go.Figure()
    fig_rsi.add_trace(go.Scatter(x=df.index, y=df['RSI'], name='RSI'))
    fig_rsi.add_hline(y=70, line_dash="dash", line_color="red")
    fig_rsi.add_hline(y=30, line_dash="dash", line_color="green")
    fig_rsi.update_layout(
        title=dict(text='Relative Strength Index (RSI)'),
        yaxis=dict(title=dict(text='RSI Value')),
        template='plotly_dark',
        height=300
    )

    fig_macd = go.Figure()
    fig_macd.add_trace(go.Scatter(x=df.index, y=df['MACD'], name='MACD'))
    fig_macd.add_trace(go.Scatter(x=df.index, y=df['Signal'], name='Signal Line'))
    fig_macd.update_layout(
        title=dict(text='Moving Average Convergence Divergence (MACD)'),
        yaxis=dict(title=dict(text='MACD Value')),
        template='plotly_dark',
        height=300
    )

    return fig_rsi, fig_macd

def plot_stock_data(df, symbol):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name='Price'
    ))
    
    fig.update_layout(
        title=dict(text=f'Stock Price Analysis for {symbol}'),
        xaxis=dict(title=dict(text='Date')),
        yaxis=dict(title=dict(text='Price (USD)')),
        showlegend=True,
        template='plotly_dark',
        height=600
    )
    
    return fig

def create_prophet_model(df):
    try:
        df = df.dropna()
        if len(df) < 2:
            return None

        prophet_df = pd.DataFrame()
        prophet_df['ds'] = df.index
        prophet_df['y'] = df['Close']
        
        if len(prophet_df.dropna()) < 2:
            return None

        model = Prophet(daily_seasonality=True)
        model.fit(prophet_df)
        
        future_dates = model.make_future_dataframe(periods=25)
        forecast = model.predict(future_dates)
        
        return forecast
    except Exception as e:
        st.error(f"Error creating prediction model: {str(e)}")
        return None

def show_stock_page(symbol):
    st.markdown("""
        <style>
        .back-button {
            color: #4CAF50;
            text-decoration: none;
            font-weight: bold;
            font-size: 16px;
        }
        </style>
    """, unsafe_allow_html=True)
    if st.button('← Back', key='back_button'):
        st.session_state.page = 'home'
        st.rerun()
    
    stock = yf.Ticker(symbol)
    info = stock.info
    
    st.title(f"{info.get('longName', symbol)} ({symbol})")
    
    is_crypto = symbol.endswith('-USD')
    
    col1, col2, col3 = st.columns(3)
    with col1:
        current_price = info.get('currentPrice', 'N/A')
        if current_price != 'N/A':
            price_display = f"${current_price:,.2f}"
        else:
            price_display = 'N/A'
            
        change_pct = info.get('regularMarketChangePercent', 0)
        if isinstance(change_pct, (int, float)):
            change_display = f"{change_pct:.2f}%"
        else:
            change_display = "N/A"
            
        st.metric("Price", price_display, change_display)
    
    with col2:
        if is_crypto:
            st.metric("Market Cap", f"${info.get('marketCap', 0)/1e9:.2f}B")
        else:
            st.metric("Sector", info.get('sector', 'N/A'))
    
    with col3:
        if is_crypto:
            st.metric("24h Volume", f"${info.get('volume24Hr', 0)/1e6:.1f}M")
        else:
            high_52w = info.get('fiftyTwoWeekHigh', 'N/A')
            if high_52w != 'N/A':
                st.metric("52W High", f"${high_52w:,.2f}")
            else:
                st.metric("52W High", "N/A")
    
    st.divider()
    
    df = fetch_stock_data(symbol)
    
    if df.empty:
        st.error("No data found for the selected stock.")
        return
    
    df = calculate_technical_indicators(df)
    forecast = create_prophet_model(df)
    
    tabs = st.tabs(["Price Analysis", "Technical Indicators", "Predictions"])
    
    with tabs[0]:
        st.markdown('<h3 style="font-size: 20px;">Price Chart & Volume</h3>', unsafe_allow_html=True)
        fig = plot_stock_data(df, symbol)
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown('<h3 style="font-size: 20px;">Key Statistics</h3>', unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if is_crypto:
                volume = info.get('volume24Hr', 0)
                if isinstance(volume, (int, float)):
                    st.metric("24h Volume", f"${volume/1e6:.1f}M")
                else:
                    st.metric("24h Volume", "N/A")
            else:
                pe_ratio = info.get('trailingPE')
                if isinstance(pe_ratio, (int, float)):
                    st.metric("P/E Ratio", f"{pe_ratio:.2f}")
                else:
                    st.metric("P/E Ratio", "N/A")
        
        with col2:
            if is_crypto:
                circ_supply = info.get('circulatingSupply', 0)
                if isinstance(circ_supply, (int, float)):
                    st.metric("Circulating Supply", f"{circ_supply:,.0f}")
                else:
                    st.metric("Circulating Supply", "N/A")
            else:
                eps = info.get('trailingEps')
                if isinstance(eps, (int, float)):
                    st.metric("EPS", f"${eps:.2f}")
                else:
                    st.metric("EPS", "N/A")
        
        with col3:
            if is_crypto:
                market_cap = info.get('marketCap', 0)
                if isinstance(market_cap, (int, float)):
                    st.metric("Market Cap", f"${market_cap/1e9:.2f}B")
                else:
                    st.metric("Market Cap", "N/A")
            else:
                dividend_yield = info.get('dividendYield', 0)
                if isinstance(dividend_yield, (int, float)):
                    st.metric("Dividend Yield", f"{dividend_yield*100:.2f}%")
                else:
                    st.metric("Dividend Yield", "N/A")
    
    with tabs[1]:
        st.markdown('<h3 style="font-size: 20px;">Technical Analysis</h3>', unsafe_allow_html=True)
        
        fig_rsi, fig_macd = plot_technical_indicators(df)
        st.plotly_chart(fig_rsi, use_container_width=True)
        st.plotly_chart(fig_macd, use_container_width=True)
    
    with tabs[2]:
        st.markdown('<h3 style="font-size: 24px;">25-Day Price Prediction</h3>', unsafe_allow_html=True)
        
        if forecast is not None:
            last_price = df['Close'].iloc[-1]
            forecast_price = forecast['yhat'].iloc[-1]
            price_change = ((forecast_price - last_price) / last_price) * 100
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Current Price", f"${last_price:.2f}")
            with col2:
                st.metric("Predicted Price", f"${forecast_price:.2f}", f"{price_change:.1f}%")
            
            fig_forecast = go.Figure()
            fig_forecast.add_trace(go.Scatter(x=df.index[-50:], y=df['Close'][-50:], name='Historical', line=dict(color='blue')))
            fig_forecast.add_trace(go.Scatter(x=forecast['ds'][-25:], y=forecast['yhat'][-25:], name='Predicted', line=dict(color='red')))
            fig_forecast.add_trace(go.Scatter(x=forecast['ds'][-25:], y=forecast['yhat_upper'][-25:], fill=None, line=dict(color='rgba(255,0,0,0.2)', width=0), showlegend=False))
            fig_forecast.add_trace(go.Scatter(x=forecast['ds'][-25:], y=forecast['yhat_lower'][-25:], fill='tonexty', line=dict(color='rgba(255,0,0,0.2)', width=0), name='Confidence Interval'))
            
            fig_forecast.update_layout(
                title=dict(text='Price Forecast'),
                xaxis=dict(title=dict(text='Date')),
                yaxis=dict(title=dict(text='Price (USD)')),
                template='plotly_dark',
                height=500
            )
            
            st.plotly_chart(fig_forecast, use_container_width=True)
        else:
            st.warning("Unable to generate predictions due to insufficient data. This could be because the stock is too new or has missing data points.")
        
        st.markdown("""
        <div style="background-color: rgba(255,255,255,0.1); padding: 15px; border-radius: 5px; margin-top: 20px;">
            <p style="margin: 0;">
                <strong>Note:</strong> Predictions are based on historical data and technical analysis. 
                Market conditions can change rapidly, and past performance does not guarantee future results.
            </p>
        </div>
        """, unsafe_allow_html=True)

def stock_page():
    show_stock_page(st.session_state.stock)