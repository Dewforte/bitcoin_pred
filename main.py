import os
import yaml
import requests
import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.models import Sequential
from logging_config import setup_logging
from dotenv import load_dotenv

# Setup logging
logger = setup_logging()

# Load environment variables
load_dotenv()

# Load configuration
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

def fetch_data(symbol, interval, start_date, end_date):
    api_key = os.getenv('API_KEY')
    api_url = f"{config['api']['url']}?symbol={symbol}&start_date={start_date}&end_date={end_date}&interval={interval}&order=asc&apikey={api_key}"
    
    try:
        response = requests.get(api_url)
        response.raise_for_status()
        data = response.json()
        return pd.DataFrame(data["values"])
    except requests.RequestException as e:
        logger.error(f"Error fetching data: {e}")
        return None

def prepare_data(data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data["close"].values.reshape(-1, 1))
    
    time_interval_train = config['model']['time_interval_train']
    prediction_interval = config['model']['prediction_interval']
    
    x_train = []
    y_train = []

    for i in range(time_interval_train, len(scaled_data) - prediction_interval):
        x_train.append(scaled_data[i - time_interval_train: i, 0])
        y_train.append(scaled_data[i + prediction_interval, 0])

    return np.array(x_train), np.array(y_train), scaler

def create_model(input_shape):
    model = Sequential()
    model.add(LSTM(128, return_sequences=True, input_shape=input_shape, activation="relu"))
    model.add(Dropout(0.4))
    model.add(LSTM(64, return_sequences=True, activation="relu"))
    model.add(Dropout(0.3))
    model.add(LSTM(32, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation="sigmoid"))

    model.compile(loss="mean_squared_error", optimizer="adam", metrics=["accuracy"])
    return model

def train_model(model, x_train, y_train):
    model.fit(x_train, y_train, epochs=config['model']['epochs'], batch_size=config['model']['batch_size'])

def predict(model, last_data, scaler):
    prediction = model.predict(last_data)
    return scaler.inverse_transform(prediction)

def main():
    logger.info("Starting crypto prediction process")
    
    symbol = os.getenv('SYMBOL')
    interval = os.getenv('INTERVAL')
    start_date = "2024-01-01 00:00:00"
    end_date = "2024-07-01 00:00:00"

    data = fetch_data(symbol, interval, start_date, end_date)
    if data is None:
        return

    x_train, y_train, scaler = prepare_data(data)
    
    model = create_model((x_train.shape[1], 1))
    train_model(model, x_train, y_train)

    # Fetch test data and make prediction
    test_data = fetch_data(symbol, interval, "2024-07-01", "2024-07-31")
    if test_data is None:
        return

    last_data = scaler.transform(test_data["close"].values[-config['model']['time_interval_train']:].reshape(-1, 1))
    last_data = np.reshape(last_data, (1, last_data.shape[0], 1))

    prediction = predict(model, last_data, scaler)
    
    logger.info(f"Prediction for next interval: {prediction[0][0]}")

if __name__ == "__main__":
    main()