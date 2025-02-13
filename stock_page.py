import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import load_model
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta

@st.cache_resource
def load_model_cached():
    return load_model('Stock-Prediction-main\stock_prediction_model.keras')

@st.cache_data
def fetch_stock_data(stock, start, end):
    return yf.download(stock, start, end)

def preprocess_data(data):
    data_train = pd.DataFrame(data.Close[0: int(len(data)*0.80)])
    data_test = pd.DataFrame(data.Close[int(len(data)*0.80): len(data)])
    scaler = MinMaxScaler(feature_range=(0,1))
    past_100_days = data_train.tail(100)
    data_test = pd.concat([past_100_days, data_test], ignore_index=True)
    data_test_scale = scaler.fit_transform(data_test)
    return data_test, data_test_scale, scaler

def create_sequences(data, seq_length):
    x, y = [], []
    for i in range(seq_length, len(data)):
        x.append(data[i-seq_length:i])
        y.append(data[i, 0])
    return np.array(x), np.array(y)

def show_stock_page():
    st.title(f'Stock Analysis: {st.session_state.stock}')

    try:
        model = load_model_cached()
    except Exception as e:
        st.error(f"Error loading the model: {e}")
        st.stop()

    col1, col2, col3 = st.columns(3)
    with col1:
        start_date = st.date_input("Start Date", datetime(2012, 1, 1))
    with col2:
        end_date = st.date_input("End Date", datetime.now())
    with col3:
        if st.button("Back to Home"):
            st.session_state.page = 'home'
            st.experimental_rerun()

    data = fetch_stock_data(st.session_state.stock, start_date, end_date)

    if data.empty:
        st.warning("No data available for the selected stock and date range.")
        st.stop()

    st.subheader('Recent Stock Data')
    st.dataframe(data.tail())

    data_test, data_test_scale, scaler = preprocess_data(data)
    
    # Visualizations
    plot_moving_averages(data)
    plot_volume_traded(data)

    # Predictions
    x, y = create_sequences(data_test_scale, 100)
    predict = model.predict(x)
    predict = predict * (1/scaler.scale_)
    y = y * (1/scaler.scale_)

    plot_predictions(data_test, y, predict)

    # 5-day prediction comparison
    compare_5_day_prediction(data, model, scaler)

    # Future prediction
    predict_next_day(data, model, scaler)

    # Model performance metrics
    display_model_performance(y, predict.flatten())

def plot_moving_averages(data):
    st.subheader('Price vs Moving Averages')
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(data.index, data.Close, label='Close Price')
    ax.plot(data.index, data.Close.rolling(50).mean(), label='MA50')
    ax.plot(data.index, data.Close.rolling(100).mean(), label='MA100')
    ax.plot(data.index, data.Close.rolling(200).mean(), label='MA200')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.legend()
    st.pyplot(fig)

def plot_volume_traded(data):
    st.subheader('Volume Traded')
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(data.index, data['Volume'], color='blue')
    ax.set_xlabel('Date')
    ax.set_ylabel('Volume')
    ax.set_title('Volume Traded')
    plt.xticks(rotation=45)
    st.pyplot(fig)

def plot_predictions(data_test, y, predict):
    st.subheader('Original Price vs Predicted Price')
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(data_test.index[100:], y, 'g', label='Original Price')
    ax.plot(data_test.index[100:], predict, 'r', label='Predicted Price')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.legend()
    st.pyplot(fig)

def compare_5_day_prediction(data, model, scaler):
    
    st.subheader('5-Day Prediction Comparison')
    
    # Ensure we have at least 100 days of data
    if len(data) < 100:
        st.warning("Not enough historical data for 5-day prediction comparison.")
        return

    last_5_days = data.tail(5)
    dates = last_5_days.index.tolist()
    actual_prices = last_5_days['Close'].tolist()
    predicted_prices = []
    
    for i in range(5):
        # Use the 100 days prior to each of the last 5 days for prediction
        last_100_days = data['Close'].iloc[-(105-i):-(5-i)].values.reshape(-1, 1)
        last_100_days_scaled = scaler.transform(last_100_days)
        X_test = np.array([last_100_days_scaled])
        pred_price = model.predict(X_test)
        pred_price = scaler.inverse_transform(pred_price)
        predicted_prices.append(pred_price[0][0])
    
    comparison_df = pd.DataFrame({
        'Date': dates,
        'Actual Close': actual_prices,
        'Predicted Close': predicted_prices
    })
    
    comparison_df['Prediction Error'] = comparison_df['Actual Close'] - comparison_df['Predicted Close']
    comparison_df['Percentage Error'] = (comparison_df['Prediction Error'] / comparison_df['Actual Close']) * 100
    
    st.dataframe(comparison_df.style.format({
        'Date': lambda x: x.strftime('%Y-%m-%d'),
        'Actual Close': '${:.2f}',
        'Predicted Close': '${:.2f}',
        'Prediction Error': '${:.2f}',
        'Percentage Error': '{:.2f}%'
    }))
    
    # Calculate and display average error metrics
    mae = np.mean(np.abs(comparison_df['Prediction Error']))
    mape = np.mean(np.abs(comparison_df['Percentage Error']))
    st.write(f"Mean Absolute Error: ${mae:.2f}")
    st.write(f"Mean Absolute Percentage Error: {mape:.2f}%")
    

def predict_next_day(data, model, scaler):
    st.subheader('Next Day Prediction')
    last_100_days = data['Close'].tail(100).values.reshape(-1, 1)
    last_100_days_scaled = scaler.transform(last_100_days)
    X_test = np.array([last_100_days_scaled])
    pred_price = model.predict(X_test)
    pred_price = scaler.inverse_transform(pred_price)

    next_date = data.index[-1] + timedelta(days=1)
    st.write(f"Predicted closing price for {next_date.date()}: ${pred_price[0][0]:.2f}")

def display_model_performance(y, predict):
    st.subheader('Model Performance')
    mse = np.mean((y - predict)**2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y - predict))

    col1, col2, col3 = st.columns(3)
    col1.metric("Mean Squared Error", f"{mse:.4f}")
    col2.metric("Root Mean Squared Error", f"{rmse:.4f}")
    col3.metric("Mean Absolute Error", f"{mae:.4f}")

    st.info("Note: These metrics are calculated on the test set. Lower values indicate better performance.")

if __name__ == "__main__":
    show_stock_page()