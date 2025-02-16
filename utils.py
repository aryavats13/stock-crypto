import yfinance as yf
import pandas as pd
import numpy as np
from ta.trend import MACD, ADXIndicator, IchimokuIndicator
from ta.momentum import RSIIndicator, StochasticOscillator, ROCIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator, VolumeWeightedAveragePrice, ChaikinMoneyFlowIndicator
import plotly.graph_objects as go
from datetime import datetime, timedelta
from prophet import Prophet
from scipy import stats

def fetch_stock_data(symbol, period='1y'):
    """Fetch stock data using yfinance"""
    try:
        stock = yf.Ticker(symbol)
        df = stock.history(period=period)
        
        # Check if we got any data
        if df.empty:
            return pd.DataFrame()
            
        # Check if we have at least 2 days of data
        if len(df) < 2:
            return pd.DataFrame()
            
        return df
    except Exception as e:
        print(f"Error fetching data for {symbol}: {str(e)}")
        return pd.DataFrame()

def calculate_technical_indicators(df):
    """Calculate technical indicators"""
    # RSI
    rsi_indicator = RSIIndicator(df['Close'])
    df['RSI'] = rsi_indicator.rsi()
    
    # MACD
    macd = MACD(df['Close'])
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()
    df['MACD_Histogram'] = macd.macd_diff()
    
    # Bollinger Bands
    bb = BollingerBands(df['Close'])
    df['BB_High'] = bb.bollinger_hband()
    df['BB_Low'] = bb.bollinger_lband()
    df['BB_Mid'] = bb.bollinger_mavg()
    df['BB_Width'] = bb.bollinger_wband()
    
    # Moving Averages
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA50'] = df['Close'].rolling(window=50).mean()
    df['MA200'] = df['Close'].rolling(window=200).mean()
    
    # Stochastic Oscillator
    stoch = StochasticOscillator(df['High'], df['Low'], df['Close'])
    df['Stoch_K'] = stoch.stoch()
    df['Stoch_D'] = stoch.stoch_signal()
    
    # Average True Range
    atr = AverageTrueRange(df['High'], df['Low'], df['Close'])
    df['ATR'] = atr.average_true_range()
    
    # On Balance Volume
    obv = OnBalanceVolumeIndicator(df['Close'], df['Volume'])
    df['OBV'] = obv.on_balance_volume()
    
    # ADX
    adx = ADXIndicator(df['High'], df['Low'], df['Close'])
    df['ADX'] = adx.adx()
    df['DI_plus'] = adx.adx_pos()
    df['DI_minus'] = adx.adx_neg()
    
    # Ichimoku Cloud
    ichimoku = IchimokuIndicator(df['High'], df['Low'])
    df['Ichimoku_Conversion'] = ichimoku.ichimoku_conversion_line()
    df['Ichimoku_Base'] = ichimoku.ichimoku_base_line()
    df['Ichimoku_A'] = ichimoku.ichimoku_a()
    df['Ichimoku_B'] = ichimoku.ichimoku_b()
    
    # VWAP
    vwap = VolumeWeightedAveragePrice(df['High'], df['Low'], df['Close'], df['Volume'])
    df['VWAP'] = vwap.volume_weighted_average_price()
    
    # Rate of Change
    roc = ROCIndicator(df['Close'])
    df['ROC'] = roc.roc()
    
    # Chaikin Money Flow
    cmf = ChaikinMoneyFlowIndicator(df['High'], df['Low'], df['Close'], df['Volume'])
    df['CMF'] = cmf.chaikin_money_flow()
    
    return df

def calculate_risk_metrics(df):
    """Calculate risk metrics"""
    returns = df['Close'].pct_change().dropna()
    
    # Beta (using market data - S&P 500)
    try:
        market = yf.download('^GSPC', start=df.index[0], end=df.index[-1])['Close']
        market_returns = market.pct_change().dropna()
        beta = stats.linregress(market_returns, returns).slope
    except:
        beta = np.nan
    
    # Volatility (annualized)
    volatility = returns.std() * np.sqrt(252)
    
    # Sharpe Ratio (assuming risk-free rate of 2%)
    rf = 0.02
    excess_returns = returns - rf/252
    sharpe = np.sqrt(252) * excess_returns.mean() / returns.std()
    
    # Maximum Drawdown
    cum_returns = (1 + returns).cumprod()
    rolling_max = cum_returns.expanding().max()
    drawdowns = cum_returns/rolling_max - 1
    max_drawdown = drawdowns.min()
    
    # Value at Risk (95% confidence)
    var_95 = np.percentile(returns, 5)
    
    return {
        'beta': beta,
        'volatility': volatility,
        'sharpe_ratio': sharpe,
        'max_drawdown': max_drawdown,
        'var_95': var_95
    }

def identify_patterns(df):
    """Identify chart patterns"""
    patterns = []
    
    # Trend Analysis
    current_price = df['Close'].iloc[-1]
    ma20 = df['MA20'].iloc[-1]
    ma50 = df['MA50'].iloc[-1]
    ma200 = df['MA200'].iloc[-1]
    
    if ma20 > ma50 and ma50 > ma200:
        patterns.append('Uptrend (Strong)')
    elif ma20 < ma50 and ma50 < ma200:
        patterns.append('Downtrend (Strong)')
    
    # Support and Resistance
    last_price = df['Close'].iloc[-1]
    price_range = df['High'].max() - df['Low'].min()
    support_zone = df['Low'].rolling(20).min().iloc[-1]
    resistance_zone = df['High'].rolling(20).max().iloc[-1]
    
    if abs(last_price - support_zone) < price_range * 0.02:
        patterns.append('Near Support')
    if abs(last_price - resistance_zone) < price_range * 0.02:
        patterns.append('Near Resistance')
    
    # Momentum Analysis
    rsi = df['RSI'].iloc[-1]
    if rsi > 70:
        patterns.append('Overbought (RSI)')
    elif rsi < 30:
        patterns.append('Oversold (RSI)')
    
    # Volume Analysis
    avg_volume = df['Volume'].mean()
    current_volume = df['Volume'].iloc[-1]
    if current_volume > 2 * avg_volume:
        patterns.append('High Volume')
    
    return patterns

def create_candlestick_chart(df, symbol):
    """Create an interactive candlestick chart with volume"""
    fig = go.Figure()
    
    # Candlestick chart
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name='OHLC'
    ))
    
    # Add volume bar chart
    colors = ['red' if row['Open'] > row['Close'] else 'green' for index, row in df.iterrows()]
    fig.add_trace(go.Bar(
        x=df.index,
        y=df['Volume'],
        name='Volume',
        marker_color=colors,
        yaxis='y2'
    ))
    
    # Update layout
    fig.update_layout(
        title=f'{symbol} Stock Price',
        yaxis_title='Stock Price',
        yaxis2=dict(
            title='Volume',
            overlaying='y',
            side='right'
        ),
        template='plotly_dark',
        xaxis_rangeslider_visible=False
    )
    
    return fig

def predict_stock_price(df, days=30):
    """Predict stock prices using Prophet"""
    # Prepare data for Prophet
    prophet_df = df.reset_index()[['Date', 'Close']].rename(
        columns={'Date': 'ds', 'Close': 'y'}
    )
    
    # Create and fit Prophet model
    model = Prophet(
        daily_seasonality=True,
        yearly_seasonality=True,
        weekly_seasonality=True,
        changepoint_prior_scale=0.05
    )
    model.fit(prophet_df)
    
    # Create future dates
    future_dates = model.make_future_dataframe(periods=days)
    
    # Make predictions
    forecast = model.predict(future_dates)
    
    return forecast

def get_stock_info(symbol):
    """Get detailed stock information"""
    stock = yf.Ticker(symbol)
    info = stock.info
    
    return {
        'name': info.get('longName', 'N/A'),
        'sector': info.get('sector', 'N/A'),
        'industry': info.get('industry', 'N/A'),
        'market_cap': info.get('marketCap', 'N/A'),
        'pe_ratio': info.get('trailingPE', 'N/A'),
        'dividend_yield': info.get('dividendYield', 'N/A'),
        'beta': info.get('beta', 'N/A'),
        '52_week_high': info.get('fiftyTwoWeekHigh', 'N/A'),
        '52_week_low': info.get('fiftyTwoWeekLow', 'N/A')
    }

def calculate_returns(df):
    """Calculate various return metrics"""
    if df.empty or len(df) < 2:
        return {
            'daily_return': 0.0,
            'total_return': 0.0,
            'current_price': 0.0
        }
    
    current_price = df['Close'].iloc[-1]
    prev_close = df['Close'].iloc[-2]
    
    daily_return = ((current_price - prev_close) / prev_close) * 100
    
    start_price = df['Close'].iloc[0]
    total_return = ((current_price - start_price) / start_price) * 100
    
    return {
        'daily_return': daily_return,
        'total_return': total_return,
        'current_price': current_price
    }
