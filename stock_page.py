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

def show_stock_page(symbol):
    # Back to home button with better styling
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
    
    # Get stock/crypto data
    stock = yf.Ticker(symbol)
    info = stock.info
    
    # Display title
    st.title(f"{info.get('longName', symbol)} ({symbol})")
    
    # Check if it's a cryptocurrency
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
    
    # Brief description
    if is_crypto:
        st.write("**About:**")
        descriptions = {
            'BTC-USD': "Bitcoin is the first decentralized cryptocurrency. It's a digital currency that enables instant payments to anyone, anywhere in the world.",
            'ETH-USD': "Ethereum is a decentralized platform that runs smart contracts. It's both a cryptocurrency and a platform for decentralized applications.",
            'DOGE-USD': "Dogecoin started as a meme-inspired cryptocurrency but has grown into a significant digital currency with a strong community.",
            'XRP-USD': "XRP is a digital asset built for payments. It enables fast, low-cost international money transfers.",
            'SOL-USD': "Solana is a high-performance blockchain platform known for its fast processing speeds and low transaction costs."
        }
        st.write(descriptions.get(symbol, "A digital currency using blockchain technology for secure, decentralized transactions."))
    
    st.divider()
    
    # Fetch stock data
    df = fetch_stock_data(symbol)
    
    if df.empty:
        st.error("No data found for the selected stock.")
        return
    
    # Calculate technical indicators
    df = calculate_technical_indicators(df)
    
    # Create Prophet forecast
    forecast = create_prophet_model(df)
    
    # Create tabs with improved fonts
    st.markdown("""
    <style>
    .stTab {
        font-size: 18px;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)
    
    tabs = st.tabs(["Price Analysis", "Technical Indicators", "Predictions"])
    
    with tabs[0]:  # Price Analysis Tab
        st.markdown('<h3 style="font-size: 20px;">Price Chart & Volume</h3>', unsafe_allow_html=True)
        fig = plot_stock_data(df, symbol)
        st.plotly_chart(fig, use_container_width=True)
        
        # Key Statistics with improved formatting
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
    
    with tabs[1]:  # Technical Indicators Tab
        st.markdown('<h3 style="font-size: 20px;">Technical Analysis</h3>', unsafe_allow_html=True)
        
        # Display RSI and MACD charts
        fig_rsi, fig_macd = plot_technical_indicators(df)
        st.plotly_chart(fig_rsi, use_container_width=True)
        st.plotly_chart(fig_macd, use_container_width=True)
    
    with tabs[2]:  # Predictions Tab
        st.markdown('<h3 style="font-size: 24px;">25-Day Price Prediction</h3>', unsafe_allow_html=True)
        
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

def stock_page():
    show_stock_page(st.session_state.stock)