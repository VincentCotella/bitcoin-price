import pandas as pd
import requests
from datetime import timedelta

# Chemin des fichiers
RAW_CSV = "data/raw/prices.csv"
PRED_CSV = "data/predictions.csv"

# 1) Charger les données réelles
df = pd.read_csv(RAW_CSV, parse_dates=["timestamp"])

# 2) Charger ou initialiser predictions.csv
try:
    preds = pd.read_csv(PRED_CSV, parse_dates=["timestamp_for_prediction"])
except FileNotFoundError:
    preds = pd.DataFrame(columns=["timestamp_for_prediction","predicted","actual"])

# 3) Mettre à jour la colonne 'actual' pour la prédiction précédente
if not preds.empty:
    last_pred_ts = preds["timestamp_for_prediction"].max()
    if last_pred_ts == df["timestamp"].iloc[-1]:
        preds.loc[preds["timestamp_for_prediction"] == last_pred_ts, "actual"] = df["close"].iloc[-1]

# 4) Construire la fenêtre de 60 dernières valeurs
last_window = df["close"].tail(60).tolist()
payload = {"window": last_window}

# 5) Appeler l’API d’inférence dans le container 'inference'
resp = requests.post("http://inference:8000/predict", json=payload)
resp.raise_for_status()
pred_value = resp.json()["prediction"]

# 6) Calculer la timestamp de la prédiction (1 min après la dernière entrée réelle)
next_ts = df["timestamp"].iloc[-1] + timedelta(minutes=1)

# 7) Ajouter la nouvelle ligne (predicted, sans actual pour l’instant)
new_row = {
    "timestamp_for_prediction": next_ts,
    "predicted": pred_value,
    "actual": None
}
preds = pd.concat([preds, pd.DataFrame([new_row])], ignore_index=True)
preds.to_csv(PRED_CSV, index=False)
print(f"[+] Logged prediction for {next_ts}: {pred_value:.2f}")
