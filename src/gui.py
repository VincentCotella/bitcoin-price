import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Bitcoin Predictor", layout="wide")
st.title("Bitcoin Price Predictor Dashboard")

# 1) Cours réels
df = pd.read_csv("data/raw/prices.csv", parse_dates=["timestamp"])
df_last = df.tail(200)

# 2) Prédictions
preds = pd.read_csv("data/predictions.csv", parse_dates=["timestamp_for_prediction"])
# On trace uniquement les prédictions dont on a un 'actual' ou non
preds = preds.sort_values("timestamp_for_prediction")

# 3) Figure combinée
fig = px.line(df_last, x="timestamp", y="close",
              labels={"close":"Prix réel USD"}, title="Prix réel vs Prédiction")
fig.add_scatter(
    x=preds["timestamp_for_prediction"],
    y=preds["predicted"],
    mode="lines+markers",
    name="Prédiction",
    line=dict(color="firebrick")
)
st.plotly_chart(fig, use_container_width=True)
