# Bitcoin Price Predictor (MLOps Local Docker Compose)

Ce projet met en place un pipeline complet MLOps – à exécuter **localement** via Docker Compose – pour :

1. **Ingestion** minute par minute des prix Bitcoin depuis l’API Binance.  
2. **Prédiction** de 1 minute dans le futur, loggée dans `data/predictions.csv`.  
3. **Réentraînement** complet (LSTM) + export TFLite toutes les 30 minutes, avec suivi MLflow.  
4. **Service d’inférence** (FastAPI + TFLite) exposant `/predict` et metrics Prometheus.  
5. **Interface Streamlit** permettant de visualiser en temps quasi réel :  
   - **Courbe bleue** = Prix réel (historique).  
   - **Courbe rouge** = Prédictions (toujours 1 minute d’avance).  
6. **Observabilité** via Prometheus, Grafana et Alertmanager.  
7. **Tracking MLflow** (runs, hyperparamètres, MAE…).  

## Prérequis

- Docker & Docker Compose  
- DVC (pour récupérer data + modèles versionnés)  
- Git (pour cloner le repo)  

## Installation & exécution en local


