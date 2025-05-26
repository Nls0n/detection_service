
FROM python:3.11-slim

WORKDIR /app


RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      libgl1 \
      libglib2.0-0 \
      gcc \
      libpq-dev \
 && rm -rf /var/lib/apt/lists/*


COPY app/requirements.txt ./app/requirements.txt
RUN pip install --no-cache-dir -r app/requirements.txt \
    && pip install --no-cache-dir opentelemetry-instrumentation opentelemetry-exporter-otlp


COPY app ./app
COPY app/data.yaml /app/data.yaml
COPY app/templates ./templates
COPY app/static    ./static
COPY app/weights   ./weights
COPY app/images ./images
COPY app/temp_uploads ./temp_uploads

# задаём адрес ML-сервиса внутри Docker-сети
ENV ML_SERVICE_URL=http://ml-service:8001

CMD ["uvicorn", "app.main:application", "--host", "0.0.0.0", "--port", "8000"]