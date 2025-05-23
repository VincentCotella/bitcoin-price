from fastapi import FastAPI, Response
from pydantic import BaseModel
import numpy as np
import tensorflow as tf
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST, start_http_server

# Charger le modèle TFLite
MODEL_PATH = "model/production_model.tflite"
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Metrics Prometheus
REQUEST_COUNT = Counter("predict_requests_total", "Total predict calls")
LATENCY = Histogram("predict_latency_seconds", "Latency of predict calls")

app = FastAPI()

class PredictIn(BaseModel):
    window: list[float]

@app.on_event("startup")
def startup_event():
    start_http_server(8001)  # exposer /metrics sur le port 8001

@app.post("/predict")
def predict(payload: PredictIn):
    REQUEST_COUNT.inc()
    import time
    start = time.time()
    data = np.array(payload.window, dtype=np.float32).reshape(1, len(payload.window), 1)
    interpreter.set_tensor(input_details[0]["index"], data)
    interpreter.invoke()
    pred = interpreter.get_tensor(output_details[0]["index"])[0][0]
    LATENCY.observe(time.time() - start)
    return {"prediction": float(pred)}

@app.get("/metrics")
def metrics():
    data = generate_latest()
    return Response(content=data, media_type=CONTENT_TYPE_LATEST)
