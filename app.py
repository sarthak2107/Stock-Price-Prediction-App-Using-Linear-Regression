import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Streamlit Page Configuration
st.set_page_config(page_title="📈 Stock Price Prediction", layout="centered")

st.title("📈 Stock Price Prediction App")
st.markdown("Enter a stock ticker to fetch historical data and predict the next day's closing price using Linear Regression.")

# User Inputs
stock = st.text_input("Enter Stock Symbol (e.g., AAPL, TSLA, INFY)", value='AAPL')
start_date = st.date_input("Start Date", pd.to_datetime("2015-01-01"))
end_date = st.date_input("End Date", pd.to_datetime("2024-12-31"))

if st.button("Predict Stock Price"):

    try:
        # Step 1: Download stock data
        df = yf.download(stock, start=start_date, end=end_date)
        df = df[['Close']]
        df['Target'] = df['Close'].shift(-1)
        df.dropna(inplace=True)

        # Step 2: Prepare training data
        X = df['Close'].values.reshape(-1, 1)  # shape (n, 1)
        y = df['Target'].values  # shape (n,)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

        # Step 3: Train model
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Step 4: Predict test data
        y_pred = model.predict(X_test)

        # Step 5: Predict next day's closing price
        last_close = df['Close'].iloc[-1].item()  # ✅ Scalar value
        next_day_input = np.array([[last_close]])  # ✅ Reshape to (1, 1)
        st.write(f"📏 Debug: next_day_input shape = {next_day_input.shape}")
        next_price = model.predict(next_day_input)[0]  # ✅ Final predicted price

        # Step 6: Visualization
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(y_test, label="Actual Price", linewidth=2)
        ax.plot(y_pred, label="Predicted Price", linestyle='--')
        ax.set_title("📉 Actual vs Predicted Closing Prices")
        ax.set_ylabel("Price")
        ax.legend()
        st.pyplot(fig)

        # Step 7: Output Prediction
        st.subheader(f"🔮 Predicted Next Closing Price for `{stock.upper()}`:")
        st.success(f"${next_price:.2f}")

        # Step 8: Show model performance
        mse = mean_squared_error(y_test, y_pred)
        st.markdown(f"📉 **Mean Squared Error (MSE):** `{mse:.4f}`")

    except Exception as e:
        st.error(f"❌ Error: {e}")
