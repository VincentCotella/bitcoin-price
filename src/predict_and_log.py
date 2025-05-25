import pandas as pd
import requests
from datetime import timedelta

# 1) Charger les données brutes
df = pd.read_csv("data/raw/prices.csv", parse_dates=["timestamp"])

# 2) Charger ou initialiser le fichier de prédictions
preds_path = "data/predictions.csv"
try:
    preds = pd.read_csv(preds_path, parse_dates=["timestamp_for_prediction"])
except FileNotFoundError:
    preds = pd.DataFrame(columns=["timestamp_for_prediction","predicted","actual"])

# 3) Mettre à jour les valeurs 'actual' pour les prédictions précédentes
last_ts = df["timestamp"].iloc[-1]
mask = preds["timestamp_for_prediction"] == last_ts
if mask.any():
    preds.loc[mask, "actual"] = df["close"].iloc[-1]

# 4) Calculer une nouvelle prédiction
window = df["close"].tail(60).tolist()
resp = requests.post(
    "http://inference:8000/predict",
    json={"window": window}
)
resp.raise_for_status()
pred = resp.json()["prediction"]

# 5) Planifier la timestamp de cette prédiction
next_ts = last_ts + timedelta(minutes=1)
new_row = {
    "timestamp_for_prediction": next_ts,
    "predicted": pred,
    "actual": None
}

# 6) Enregistrer
preds = pd.concat([preds, pd.DataFrame([new_row])], ignore_index=True)
preds.to_csv(preds_path, index=False)
print(f"[+] Logged prediction for {next_ts}: {pred:.2f}")
