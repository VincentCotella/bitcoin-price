import requests
import pandas as pd
from datetime import datetime

# Configuration de l'API
API_URL = "https://api.binance.com/api/v3/klines"
PARAMS = {
    "symbol": "BTCUSDT",
    "interval": "1m",
    "limit": 1000
}
OUTPUT_CSV = "data/raw/prices.csv"

def fetch_prices():
    """Interroge l'API Binance et retourne un DataFrame pandas formaté."""
    resp = requests.get(API_URL, params=PARAMS)
    resp.raise_for_status()
    data = resp.json()

    df = pd.DataFrame(data, columns=[
        "open_time","open","high","low","close","volume",
        "close_time","quote_asset_volume","num_trades",
        "taker_buy_base","taker_buy_quote","ignore"
    ])

    # Garder uniquement ce qui nous intéresse
    df = df[["open_time","open","high","low","close","volume"]]
    df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms")
    # Conversion en float
    for col in ["open","high","low","close","volume"]:
        df[col] = df[col].astype(float)

    return df[["timestamp","open","high","low","close","volume"]]

if __name__ == "__main__":
    df = fetch_prices()
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"[+] Ingestion réussie : {len(df)} lignes écrites dans {OUTPUT_CSV}")
