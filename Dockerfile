# Dockerfile

# 1) Builder : installer TensorFlow et générer .tflite
FROM python:3.10-slim AS builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir tensorflow
COPY src/export_model_to_tflite.py ./export_model_to_tflite.py
COPY model/production_model.h5 ./model/production_model.h5
RUN python export_model_to_tflite.py

# 2) Runtime : service FastAPI + TFLite
FROM python:3.10-slim AS runtime
WORKDIR /app
RUN mkdir -p model
COPY --from=builder /app/model/production_model.tflite ./model/production_model.tflite
COPY src/serve.py ./serve.py
RUN pip install --no-cache-dir fastapi uvicorn[tls] prometheus_client tensorflow
EXPOSE 8000 8001
CMD ["uvicorn", "serve:app", "--host", "0.0.0.0", "--port", "8000"]

