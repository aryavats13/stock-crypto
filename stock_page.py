import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from prophet import Prophet
import pandas as pd
from datetime import datetime, timedelta
import ta

def fetch_stock_data(symbol, period='2y'):
    try:
        stock = yf.Ticker(symbol)
        df = stock.history(period=period)
        if df.empty:
            raise ValueError(f"No data found for symbol {symbol}")
        return df
    except Exception as e:
        st.error(f"Error fetching data for {symbol}: {str(e)}")
        return pd.DataFrame()

def calculate_technical_indicators(df):
    # RSI
    df['RSI'] = ta.momentum.rsi(df['Close'])
    
    # MACD
    macd = ta.trend.MACD(df['Close'])
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()
    
    # Moving Averages
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA50'] = df['Close'].rolling(window=50).mean()
    df['MA200'] = df['Close'].rolling(window=200).mean()
    
    # Bollinger Bands
    bollinger = ta.volatility.BollingerBands(df['Close'])
    df['BB_High'] = bollinger.bollinger_hband()
    df['BB_Low'] = bollinger.bollinger_lband()
    df['BB_Mid'] = bollinger.bollinger_mavg()
    
    return df

def plot_stock_data(df, symbol):
    fig = go.Figure()
    
    # Add candlestick
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name='OHLC',
            increasing_line_color='#26a69a',
            decreasing_line_color='#ef5350'
        )
    )
    
    # Add Moving averages
    fig.add_trace(go.Scatter(x=df.index, y=df['MA20'], name='MA20', 
                            line=dict(color='#AB47BC', width=1.5)))
    fig.add_trace(go.Scatter(x=df.index, y=df['MA50'], name='MA50', 
                            line=dict(color='#7E57C2', width=1.5)))
    
    # Add volume bars
    colors = ['#ef5350' if row['Open'] - row['Close'] >= 0 else '#26a69a' 
              for index, row in df.iterrows()]
    fig.add_trace(
        go.Bar(
            x=df.index,
            y=df['Volume'],
            name='Volume',
            marker_color=colors,
            marker_line_color='rgb(0,0,0)',
            marker_line_width=0.5,
            opacity=0.7,
            yaxis='y2'
        )
    )
    
    # Add Bollinger Bands
    fig.add_trace(go.Scatter(x=df.index, y=df['BB_High'], name='BB High', 
                            line=dict(color='rgba(236, 64, 122, 0.3)', dash='dash')))
    fig.add_trace(go.Scatter(x=df.index, y=df['BB_Low'], name='BB Low',
                            line=dict(color='rgba(236, 64, 122, 0.3)', dash='dash')))
    fig.add_trace(go.Scatter(x=df.index, y=df['BB_Mid'], name='BB Mid',
                            line=dict(color='rgba(236, 64, 122, 0.8)', dash='dash')))
    
    # Update layout with improved fonts and styling
    fig.update_layout(
        title=dict(
            text=f'Stock Price Analysis for {symbol}'
        ),
        xaxis=dict(
            title=dict(
                text='Date',
                font=dict(size=14)
            ),
            tickfont=dict(size=12)
        ),
        yaxis=dict(
            title=dict(
                text='Price (USD)',
                font=dict(size=14)
            ),
            tickfont=dict(size=12)
        ),
        showlegend=True,
        template='plotly_dark',
        height=600
    )
    
    return fig

def plot_technical_indicators(df):
    # RSI Plot
    fig_rsi = go.Figure()
    fig_rsi.add_trace(go.Scatter(x=df.index, y=df['RSI'], name='RSI',
                                line=dict(color='#7E57C2', width=2)))
    fig_rsi.add_hline(y=70, line_dash="dash", line_color="#ef5350")
    fig_rsi.add_hline(y=30, line_dash="dash", line_color="#26a69a")
    fig_rsi.update_layout(
        title='Relative Strength Index (RSI)',
        height=300,
        template='plotly_dark',
        showlegend=True,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis_title='Date',
        yaxis_title='RSI Value',
        margin=dict(l=50, r=50, t=80, b=50),
        font=dict(color='white')
    )
    
    # MACD Plot
    fig_macd = go.Figure()
    fig_macd.add_trace(go.Scatter(x=df.index, y=df['MACD'], name='MACD',
                                 line=dict(color='#AB47BC', width=2)))
    fig_macd.add_trace(go.Scatter(x=df.index, y=df['MACD_Signal'], name='Signal Line',
                                 line=dict(color='#26a69a', width=2)))
    fig_macd.update_layout(
        title='Moving Average Convergence Divergence (MACD)',
        height=300,
        template='plotly_dark',
        showlegend=True,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis_title='Date',
        yaxis_title='MACD Value',
        margin=dict(l=50, r=50, t=80, b=50),
        font=dict(color='white')
    )
    
    return fig_rsi, fig_macd

def create_prophet_model(df):
    # Prepare data for Prophet
    prophet_df = df.reset_index()[['Date', 'Close']]
    # Remove timezone from dates
    prophet_df['Date'] = prophet_df['Date'].dt.tz_localize(None)
    prophet_df.columns = ['ds', 'y']
    
    # Create and fit the model
    model = Prophet(daily_seasonality=True)
    model.fit(prophet_df)
    
    # Create future dates for 25 days
    future_dates = model.make_future_dataframe(periods=25)
    forecast = model.predict(future_dates)
    
    return forecast

def stock_page():
    # Set page title with larger font
    st.markdown('<h1 style="font-size: 36px; color: white;">Stock Analysis</h1>', unsafe_allow_html=True)
    
    # Fetch stock data
    df = fetch_stock_data(st.session_state.stock)
    
    if df.empty:
        st.error("No data found for the selected stock.")
        return
    
    # Calculate technical indicators
    df = calculate_technical_indicators(df)
    
    # Create Prophet forecast
    forecast = create_prophet_model(df)
    
    # Display stock info
    stock = yf.Ticker(st.session_state.stock)
    info = stock.info
    
    # Company Info Section with improved fonts
    st.markdown('<h2 style="font-size: 24px; color: white;">Company Information</h2>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Current Price", f"${info.get('currentPrice', 'N/A')}", 
                 f"{info.get('regularMarketChangePercent', 0):.2f}%")
    with col2:
        st.metric("Market Cap", f"${info.get('marketCap', 0)/1e9:.2f}B")
    with col3:
        st.metric("52 Week High", f"${info.get('fiftyTwoWeekHigh', 'N/A')}")
    
    # Create tabs with improved fonts
    st.markdown("""
    <style>
    .stTab {
        font-size: 18px;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)
    
    tabs = st.tabs([" Price Analysis", " Technical Indicators", "Predictions"])
    
    with tabs[0]:  # Price Analysis Tab
        st.markdown('<h3 style="font-size: 20px; color: white;">Price Chart & Volume</h3>', unsafe_allow_html=True)
        fig = plot_stock_data(df, st.session_state.stock)
        st.plotly_chart(fig, use_container_width=True)
        
        # Key Statistics with improved fonts
        st.markdown('<h3 style="font-size: 20px; color: white;">Key Statistics</h3>', unsafe_allow_html=True)
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("P/E Ratio", f"{info.get('trailingPE', 'N/A'):.2f}")
        with col2:
            st.metric("Beta", f"{info.get('beta', 'N/A'):.2f}")
        with col3:
            st.metric("Volume", f"{info.get('volume', 'N/A'):,}")
        with col4:
            st.metric("Avg Volume", f"{info.get('averageVolume', 'N/A'):,}")
    
    with tabs[1]:  # Technical Indicators Tab
        st.markdown('<h3 style="font-size: 20px; color: white;">Technical Analysis</h3>', unsafe_allow_html=True)
        
        # Display RSI and MACD charts
        fig_rsi, fig_macd = plot_technical_indicators(df)
        st.plotly_chart(fig_rsi, use_container_width=True)
        st.plotly_chart(fig_macd, use_container_width=True)
    
    with tabs[2]:  # Predictions Tab
        st.markdown('<h3 style="font-size: 24px; color: white;">25-Day Price Prediction</h3>', unsafe_allow_html=True)
        
        # Current price and stats
        last_price = df['Close'].iloc[-1]
        predicted_price = forecast['yhat'].iloc[-1]
        upper_price = forecast['yhat_upper'].iloc[-1]
        lower_price = forecast['yhat_lower'].iloc[-1]
        price_change = ((predicted_price - last_price) / last_price) * 100
        
        # Display predictions in a clean format
        st.markdown("""
        <style>
        .prediction-box {
            background-color: rgba(0,0,0,0.5);
            padding: 20px;
            border-radius: 10px;
            margin: 10px 0;
            border: 1px solid rgba(255,255,255,0.1);
        }
        .price-text {
            color: white;
            font-size: 18px;
            margin: 10px 0;
        }
        .highlight {
            color: #00ff00;
            font-weight: bold;
        }
        .warning {
            color: #ff9800;
            font-size: 16px;
            font-style: italic;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Current Price Box
        st.markdown(f"""
        <div class="prediction-box">
            <h4 style="color: white; font-size: 20px;">Current Status</h4>
            <p class="price-text">Current Price: <span class="highlight">${last_price:.2f}</span></p>
            <p class="price-text">Trading Volume: <span class="highlight">{df['Volume'].iloc[-1]:,.0f}</span></p>
        </div>
        """, unsafe_allow_html=True)
        
        # Prediction Box
        st.markdown(f"""
        <div class="prediction-box">
            <h4 style="color: white; font-size: 20px;">25-Day Forecast</h4>
            <p class="price-text">Predicted Price: <span class="highlight">${predicted_price:.2f}</span></p>
            <p class="price-text">Expected Change: <span class="highlight">{price_change:+.2f}%</span></p>
            <p class="price-text">Price Range:</p>
            <ul class="price-text">
                <li>Upper Estimate: ${upper_price:.2f}</li>
                <li>Lower Estimate: ${lower_price:.2f}</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Analysis Box
        trend = "Upward" if price_change > 0 else "Downward"
        confidence_range = ((upper_price - lower_price) / predicted_price) * 100
        
        st.markdown(f"""
        <div class="prediction-box">
            <h4 style="color: white; font-size: 20px;">Analysis</h4>
            <p class="price-text">• Predicted Trend: <span class="highlight">{trend}</span></p>
            <p class="price-text">• Confidence Range: <span class="highlight">±{confidence_range:.1f}%</span></p>
            <p class="price-text">• Based on historical patterns and market indicators, the stock shows:</p>
            <ul class="price-text">
                <li>{'Strong' if abs(price_change) > 10 else 'Moderate'} {trend.lower()} momentum</li>
                <li>{'High' if confidence_range > 20 else 'Moderate'} price volatility expected</li>
            </ul>
            <p class="warning">Note: These predictions are based on historical data and should not be the sole basis for investment decisions.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Disclaimer with improved fonts
    st.markdown("""
    <div style='background-color: rgba(0,0,0,0.5); padding: 20px; border-radius: 5px; margin-top: 30px;'>
        <p style='color: white; font-size: 16px; font-family: Arial, sans-serif;'>
            This analysis is based on historical data and technical indicators. 
            Past performance is not indicative of future results. 
            Please conduct your own research before making investment decisions.
        </p>
    </div>
    """, unsafe_allow_html=True)