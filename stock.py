import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go

# Set up Streamlit title and sidebar
st.title('STOCK PREDICTION PROFIT PULSE APP')
st.sidebar.subheader('Enter Stock and Date Information')

# Accept user input for stock symbol, start date, and end date
stock_symbol = st.sidebar.text_input("Enter stock symbol (e.g., AAPL)", 'AAPL')
start_date = st.sidebar.date_input("Enter start date", pd.to_datetime('2022-01-01'))
end_date = st.sidebar.date_input("Enter end date", pd.to_datetime('2022-12-31'))

# Fetch historical data from Yahoo Finance
def load_data(symbol, start, end):
    try:
        data = yf.download(symbol, start=start, end=end)
        return data
    except Exception as e:
        st.error(f"Error fetching data for {symbol}: {e}")
        return None

# Function to display historical data
def display_historical_data(data):
    st.subheader('Historical Data')
    st.write(data)

# Load historical data
data_load_state = st.text('Loading historical data...')
historical_data = load_data(stock_symbol, start_date, end_date)

if historical_data is not None and not historical_data.empty:
    data_load_state.text('Loading historical data... done!')

    # Display historical data
    display_historical_data(historical_data)

    # Predict stock prices
    st.subheader('Stock Price Prediction')

    # Classification problem: buy(+1) or sell(-1) the stock
    historical_data['Open-Close'] = historical_data['Open'] - historical_data['Close']
    historical_data['High-Low'] = historical_data['High'] - historical_data['Low']
    historical_data = historical_data.dropna()

    # Input features to predict whether the stock should go up (+1) or down (-1)
    X = historical_data[['Open-Close', 'High-Low']]

    # Intention to store +1 for the up signal and -1 for the down signal.
    Y = np.where(historical_data['Close'].shift(-1) > historical_data['Close'], 1, -1)

    # Split the dataset into training and testing sets
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

    # Train the model
    model = LinearRegression()
    model.fit(X_train, Y_train)

    # Predictions
    predictions = model.predict(X_test)

    # Display actual and predicted prices
    prediction_df = pd.DataFrame({'Actual': Y_test, 'Predicted': predictions}, index=X_test.index)
    st.write(prediction_df)

    # Candlestick Chart
    st.subheader('Candlestick Chart')

    # Create Candlestick Chart for Actual Data
    fig_actual = go.Figure(data=[go.Candlestick(x=historical_data.index,
                                                open=historical_data['Open'],
                                                high=historical_data['High'],
                                                low=historical_data['Low'],
                                                close=historical_data['Close'],
                                                name='Actual Data')])

    fig_actual.update_xaxes(rangeslider_visible=True)
    fig_actual.update_layout(title='Actual Candlestick Chart', xaxis_title='Date', yaxis_title='Stock Price')

    # Show the actual candlestick chart
    st.plotly_chart(fig_actual)

    # Create Candlestick Chart for Predicted Data
    predicted_data = historical_data.iloc[-len(predictions):]  # Selecting data for predicted period
    predicted_data['Close_Predicted'] = predicted_data['Close'] + predictions  # Adding predicted changes to close

    fig_predicted = go.Figure(data=[go.Candlestick(x=predicted_data.index,
                                                   open=predicted_data['Open'],
                                                   high=predicted_data['High'],
                                                   low=predicted_data['Low'],
                                                   close=predicted_data['Close_Predicted'],
                                                   name='Predicted Data')])

    fig_predicted.update_xaxes(rangeslider_visible=True)
    fig_predicted.update_layout(title='Predicted Candlestick Chart', xaxis_title='Date', yaxis_title='Stock Price')

    # Show the predicted candlestick chart
    st.plotly_chart(fig_predicted)

else:
    st.warning("No data available for the specified stock symbol and date range. Please check the symbol and date range.")



