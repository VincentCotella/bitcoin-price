import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Input, LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import os

# Chemins de données et sortie
DATA_CSV = "data/raw/prices.csv"
OUTPUT_DIR = "model"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "quick_mae.txt")

def quick_train():
    # 1. Chargement des données
    df = pd.read_csv(DATA_CSV, parse_dates=["timestamp"])
    series = df["close"].values.reshape(-1, 1)

    # 2. Normalisation
    scaler = MinMaxScaler()
    data = scaler.fit_transform(series)

    # 3. Création des séquences temporelles
    seq_len = 10
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i : i + seq_len])
        y.append(data[i + seq_len])
    X, y = np.array(X), np.array(y)

    # 4. Définition du modèle
    model = Sequential([
        Input(shape=(seq_len, 1)),  # évite l'avertissement d'input_shape
        LSTM(16),
        Dense(1)
    ])
    model.compile(loss="mse", optimizer="adam")

    # 5. Entraînement rapide
    history = model.fit(X, y, epochs=3, batch_size=16, verbose=0)
    mae = float(history.history["loss"][-1])

    # 6. Sauvegarde du MAE
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        f.write(f"{mae:.6f}")

    print(f"[+] Quick train terminé – MAE: {mae:.6f}")

if __name__ == "__main__":
    quick_train()
