# Stage 1: Builder — устанавливаем зависимости
FROM python:3.9-slim as builder

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Stage 2: Runtime — минимальный образ для продакшена
FROM python:3.9-slim

WORKDIR /app

# Копируем только установленные пакеты из builder (для уменьшения размера)
COPY --from=builder /usr/local/lib/python3.9/site-packages /usr/local/lib/python3.9/site-packages

# Копируем код API
COPY src/api/ ./api/

# Копируем ONNX модель (используем обычную)
COPY models/model.onnx ./models/model.onnx

# Дополнительные пакеты, если их нет в requirements.txt
RUN pip install --no-cache-dir onnxruntime uvicorn fastapi

EXPOSE 8000

# Запуск API (предполагаю, что файл src/api/app.py с переменной app)
CMD ["python", "-m", "uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]