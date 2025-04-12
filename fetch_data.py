import yfinance as yf
import os

def load_data():
    df = yf.download('AAPL', start='2023-01-01', end='2024-01-01', progress=False)
    df = df[['Close']]  # Only keep 'Close' column
    os.makedirs('data', exist_ok=True)
    df.to_csv('data/aapl.csv')
    print("Data fetched and saved as aapl.csv.")

if __name__ == "__main__":
    load_data()
