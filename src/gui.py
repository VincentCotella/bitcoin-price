import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Dashboard Bitcoin Predictor", layout="wide")
st.title("Dashboard Bitcoin Price Predictor")

# Charger les prix réels
df = pd.read_csv("data/raw/prices.csv", parse_dates=["timestamp"])
df_last = df.tail(200)

# Charger les prédictions
try:
    preds = pd.read_csv("data/predictions.csv", parse_dates=["timestamp_for_prediction"])
    preds = preds.sort_values("timestamp_for_prediction")
except FileNotFoundError:
    preds = pd.DataFrame(columns=["timestamp_for_prediction","predicted","actual"])

# Tracer le prix réel
fig = px.line(df_last, x="timestamp", y="close",
              labels={"close":"Prix réel (USD)"}, title="Prix réel vs Prédiction (1 min d’avance)")
# Ajouter la courbe prédite
if not preds.empty:
    fig.add_scatter(
        x=preds["timestamp_for_prediction"],
        y=preds["predicted"],
        mode="lines+markers",
        name="Prédiction (USD)",
        line=dict(color="crimson")
    )
st.plotly_chart(fig, use_container_width=True)
