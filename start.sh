#!/usr/bin/env bash

# start.sh: запуск ML-сервиса и фронтенд-приложения для вашего репозитория

# Определяем корень проекта (контейнер APPLICATION)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -d "$SCRIPT_DIR/APPLICATION" ]; then
  PROJECT_ROOT="$SCRIPT_DIR/APPLICATION"
else
  PROJECT_ROOT="$SCRIPT_DIR"
fi

# Устанавливаем PYTHONPATH для обоих модулей
export PYTHONPATH="$PROJECT_ROOT"

# Запуск ML-сервиса
echo "[INFO] Starting ML service at http://127.0.0.1:8080..."
cd "$PROJECT_ROOT"
uvicorn predict_service.ml_service:app --reload --host 127.0.0.1 --port 8080 &
ML_PID=$!

# Запуск фронтенд-приложения
echo "[INFO] Starting frontend service at http://127.0.0.1:8000..."
cd "$PROJECT_ROOT/app"
uvicorn main:application --reload --host 127.0.0.1 --port 8000 &
FRONTEND_PID=$!

# Ожидаем завершения процессов
wait $ML_PID $FRONTEND_PID
