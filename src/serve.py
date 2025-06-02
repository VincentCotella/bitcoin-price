from fastapi import FastAPI, Response
from pydantic import BaseModel
import numpy as np
import tensorflow as tf
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST, start_http_server
from contextlib import asynccontextmanager

# Charger le modèle TFLite
interpreter = tf.lite.Interpreter(model_path="model/production_model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Compteurs Prometheus
REQUEST_COUNT = Counter("predict_requests_total", "Total predict calls")
LATENCY = Histogram("predict_latency_seconds", "Latency of predict")

# Créer l’application FastAPI
app = FastAPI()

# Définition du modèle d’entrée
class PredictIn(BaseModel):
    window: list[float]  # Liste de 60 valeurs de prix

# Utiliser Lifespan pour démarrer le serveur de metrics Prometheus
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Au tout début, lancer le serveur de metrics sur le port 8001
    start_http_server(8001)
    yield
    # (Pas d’action spécifique à la fermeture)

app.router.lifespan_context = lifespan

@app.post("/predict")
def predict(payload: PredictIn):
    REQUEST_COUNT.inc()
    import time
    start = time.time()

    arr = np.array(payload.window, dtype=np.float32).reshape(1, len(payload.window), 1)
    interpreter.set_tensor(input_details[0]['index'], arr)
    interpreter.invoke()
    pred = interpreter.get_tensor(output_details[0]['index'])[0][0]

    LATENCY.observe(time.time() - start)
    return {"prediction": float(pred)}

@app.get("/metrics")
def metrics():
    data = generate_latest()
    return Response(content=data, media_type=CONTENT_TYPE_LATEST)
