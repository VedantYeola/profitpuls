import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go
from pages.login import authenticate_user



# Set up Streamlit title and sidebar
st.title('Stock Price Prediction')
st.sidebar.subheader('Enter Stock and Date Information')

# Accept user input for stock symbol, start date, and end date
stock_symbol = st.sidebar.text_input("Enter stock symbol", 'AAPL')
start_date = st.sidebar.date_input("Enter start date", pd.to_datetime('2024-01-01'))
end_date = st.sidebar.date_input("Enter end date", pd.to_datetime('2024-1-31'))

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
            data_load_state.text(f'Loading historical data for {stock_symbol} ... done!')

            display_historical_data(historical_data)

            st.subheader('Stock Price Prediction')

            historical_data['Open-Close'] = historical_data['Open'] - historical_data['Close']
            historical_data['High-Low'] = historical_data['High'] - historical_data['Low']
            historical_data = historical_data.dropna()

            X = historical_data[['Open-Close', 'High-Low']]

            Y = np.where(historical_data['Close'].shift(-1) > historical_data['Close'], 1, -1)

            X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

            model = LinearRegression()
            model.fit(X_train, Y_train)

            predictions = model.predict(X_test)

            prediction_df = pd.DataFrame({'Actual': Y_test, 'Predicted': predictions}, index=X_test.index)
            st.write(prediction_df)

            st.subheader('Stock Price Prediction')

            historical_data['Open-Close'] = historical_data['Open'] - historical_data['Close']
            historical_data['High-Low'] = historical_data['High'] - historical_data['Low']
            historical_data = historical_data.dropna()

            X = historical_data[['Open-Close', 'High-Low']]

            Y = np.where(historical_data['Close'].shift(-1) > historical_data['Close'], 1, -1)

            X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

            model = LinearRegression()
            model.fit(X_train, Y_train)

            predictions = model.predict(X_test)

            st.line_chart(pd.DataFrame({'Actual': Y_test, 'Predicted': predictions}, index=X_test.index))

            st.subheader('Candlestick Chart with Predicted Prices')

            fig = go.Figure(data=[
                go.Candlestick(x=historical_data.index,
                               open=historical_data['Open'],
                               high=historical_data['High'],
                               low=historical_data['Low'],
                               close=historical_data['Close'],
                               name='Historical Prices'),

                go.Candlestick(x=X_test.index,
                               open=historical_data.loc[X_test.index, 'Open'],
                               high=historical_data.loc[X_test.index, 'High'],
                               low=historical_data.loc[X_test.index, 'Low'],
                               close=predictions,
                               name='Predicted Prices',
                               increasing_line_color='aqua',  # Set the color for increasing (bullish) candles
                               decreasing_line_color='violet'  # Set the color for decreasing (bearish) candles
                           )
            ])

            fig.update_xaxes(rangeslider_visible=True)

            fig.update_layout(title='Candlestick Chart with Predicted Prices', xaxis_title='Date', yaxis_title='Stock Price')

            st.plotly_chart(fig)

            st.subheader('Candlestick Chart')
            fig = go.Figure(data=[go.Candlestick(x=historical_data.index,
                                                 open=historical_data['Open'],
                                                 high=historical_data['High'],
                                                 low=historical_data['Low'],
                                                 close=historical_data['Close'])])

            fig.update_xaxes(rangeslider_visible=True)
            fig.update_layout(title='Candlestick Chart', xaxis_title='Date', yaxis_title='Stock Price')

            st.plotly_chart(fig)

else:
            st.warning("No data available for the specified stock symbol and date range. Please check the symbol and date range.")

# if __name__ == "__main__":
#     main()