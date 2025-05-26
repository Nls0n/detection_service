FROM python:3.11-slim

WORKDIR /app


RUN apt-get update \
 && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*


COPY predict_service/requirements.txt ./predict_service/requirements.txt
RUN pip install --no-cache-dir -r predict_service/requirements.txt \
    && pip install --no-cache-dir opentelemetry-instrumentation opentelemetry-exporter-otlp


COPY predict_service ./predict_service
COPY app/weights ./app/weights

ENV OTEL_EXPORTER_OTLP_ENDPOINT=http://otel-collector:4317 \
    OTEL_SERVICE_NAME=ml-service

EXPOSE 8001

CMD ["opentelemetry-instrument", "uvicorn", "predict_service.ml_service:app", \
     "--host", "0.0.0.0", "--port", "8001"]