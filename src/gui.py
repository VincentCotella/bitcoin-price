# src/gui.py
import streamlit as st
import pandas as pd
import requests
import plotly.express as px

PI_API = "http://localhost:8000/predict"

st.set_page_config(page_title="Bitcoin Predictor", layout="wide")

st.title("Dashboard Bitcoin Price Predictor")

# 1) Afficher les cours bruts
df = pd.read_csv("data/raw/prices.csv", parse_dates=["timestamp"])
df_last = df.tail(200)
fig = px.line(df_last, x="timestamp", y="close", title="Cours Bitcoin (dernières 200 mesures)")
st.plotly_chart(fig, use_container_width=True)

# 2) Prédiction en temps réel
st.markdown("### Prédiction en direct")
window = df["close"].tail(60).tolist()  # 60 dernières valeurs
if st.button("Prédire le prochain cours"):
    payload = {"window": window}
    with st.spinner("Calcul en cours..."):
        resp = requests.post(PI_API, json=payload)
    if resp.ok:
        pred = resp.json().get("prediction")
        st.metric("Prix prédit (USD)", f"{pred:.2f}")
    else:
        st.error(f"Erreur API : {resp.status_code}")
