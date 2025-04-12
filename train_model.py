import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import save_model

# Fetch stock data for AAPL (adjust the dates as needed)
df = yf.download('AAPL', start='2023-01-01', end='2024-01-01')
df = df[['Close']]  # Use only the 'Close' price for training
df.dropna(inplace=True)  # Remove any rows with missing values

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df)

# Prepare the data for training (using 60 previous days to predict the next day)
X_train, y_train = [], []
for i in range(60, len(scaled_data)):
    X_train.append(scaled_data[i - 60:i, 0])
    y_train.append(scaled_data[i, 0])

X_train, y_train = np.array(X_train), np.array(y_train)

# Reshape X_train for LSTM (samples, time steps, features)
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))

# Build the LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dense(units=1))  # Output layer (single value)

# Compile and train the model
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, epochs=10, batch_size=32)

# Save the trained model
model.save('saved_model/saved_lstm_model.h5')  # Save the model to the saved_model folder

print("Model trained and saved successfully!")
