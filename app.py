from flask import Flask, render_template, request
import os
import numpy as np
import pandas as pd
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__, template_folder='templates')

MODEL_PATH = 'saved_model/saved_lstm_model.h5'  # Path to the saved LSTM model
DATA_PATH = 'data/aapl.csv'         # Path to your stock price data file

# Load the dataset (This example uses 'Close' column)
def load_data():
    df = pd.read_csv(DATA_PATH)
    df = df[['Close']]  # Select only the 'Close' column
    df.dropna(inplace=True)  # Remove any missing values just in case
    return df

# Load the saved LSTM model once at the start of the app
model = load_model(MODEL_PATH)

# Preprocess data and return scaled training sets
def preprocess_data(df):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df)

    X_train, y_train = [], []
    for i in range(60, len(scaled_data) - 60):
        X_train.append(scaled_data[i - 60:i, 0])
        y_train.append(scaled_data[i, 0])

    X_train = np.array(X_train)
    y_train = np.array(y_train)

    if X_train.ndim == 2:
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))

    return X_train, y_train, scaler

# Flask route to handle predictions
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        days = request.form['days']
        
        try:
            days = int(days)  # Convert the input into an integer
        except ValueError:
            days = 0  # If the input is invalid, default to 0
        
        # Prepare the input for the LSTM model (days represents future days for prediction)
        df = load_data()  # Load historical stock data
        _, _, scaler = preprocess_data(df)  # Get the preprocessed data and scaler

        # Create a sequence from the last 60 days to predict the future price
        last_60_days = df[-60:].values
        last_60_days_scaled = scaler.transform(last_60_days)

        X_test = last_60_days_scaled.reshape((1, 60, 1))  # Shape input for LSTM (1 sample, 60 timesteps, 1 feature)

        # Predict the future stock price
        predicted_price = model.predict(X_test)
        predicted_price = scaler.inverse_transform(predicted_price)  # Transform the prediction back to original scale

        # Convert predicted price to a float for display
        price = float(predicted_price[0][0])

        return render_template('index.html', price=price)

    return render_template('index.html', price=None)

if __name__ == '__main__':
    print("Flask app is starting...")
    app.run(debug=True)
