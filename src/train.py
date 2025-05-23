import pandas as pd
import numpy as np
import os
import mlflow
import mlflow.keras
from keras.models import Sequential
from keras.layers import Input, LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

# Chemins
DATA_CSV = "data/raw/prices.csv"
OUTPUT_DIR = "model"
H5_MODEL = os.path.join(OUTPUT_DIR, "production_model.h5")

# Configuration MLflow
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("bitcoin-price-pi")

def train_full():
    # 1. Chargement des données
    df = pd.read_csv(DATA_CSV, parse_dates=["timestamp"])
    series = df["close"].values.reshape(-1, 1)

    # 2. Normalisation
    scaler = MinMaxScaler()
    data = scaler.fit_transform(series)

    # 3. Préparation des séquences (seq_len=60)
    seq_len = 60
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i : i + seq_len])
        y.append(data[i + seq_len])
    X, y = np.array(X), np.array(y)

    # 4. Entraînement et tracking
    with mlflow.start_run():
        model = Sequential([
            Input(shape=(seq_len, 1)),
            LSTM(32),
            Dense(1)
        ])
        model.compile(loss="mse", optimizer="adam")
        history = model.fit(X, y, epochs=10, batch_size=32, verbose=1)
        mae = float(history.history["loss"][-1])

        # Logging des métriques et du modèle
        mlflow.log_metric("mae", mae)
        mlflow.keras.log_model(model, artifact_path="model")

        # Sauvegarde du modèle pour export TFLite
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        model.save(H5_MODEL)

        print(f"[+] Training done – MAE: {mae:.6f}")

if __name__ == "__main__":
    train_full()
