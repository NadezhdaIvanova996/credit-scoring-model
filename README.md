## Credit Scoring Model
## Описание
Проект для предсказания дефолта по кредитам с использованием машинного обучения. Реализована модель на основе GradientBoosting с метриками: AUC=0.76, precision=0.69, recall=0.31, f1=0.43.
Итоговый проект по курсу «Автоматизация процессов доставки и развертывания моделей МО» — полный MLOps цикл.
## Структура проекта

src/: Код (данные, пайплайн, обучение, API).
data/: Обработанные данные (версионируются DVC).
tests/: Тесты для кода.
models/: Сохранённые модели (pkl, ONNX, quantized).
mlruns/: Данные экспериментов MLflow.
.github/workflows/: Конфигурация CI/CD.
Dockerfile: Multi-stage контейнер для API с ONNX моделью.
scripts/: Скрипты для конвертации, оптимизации, мониторинга дрифта.
infrastructure/: Terraform для Yandex Cloud (VPC, subnets, Managed Kubernetes).
deployment/: Kubernetes manifests и kubeconfig.
monitoring/: Конфиги для мониторинга (Prometheus, Grafana, alerts).
airflow/dags/: DAG для переобучения модели.

## Установка

Установите Docker: docker.com.
Клонируйте репозиторий: git clone https://github.com/NadezhdaIvanova996/credit-scoring-model.git.
Перейдите в папку: cd credit-scoring-model.
Установите зависимости: pip install -r requirements.txt.

## Запуск локально

Соберите Docker-образ: docker build -t credit-scoring-api:latest ..
Запустите контейнер: docker run -p 8000:8000 credit-scoring-api:latest.
Проверьте API: http://localhost:8000/docs (Swagger UI).

## Тестирование

Запустите тесты: PYTHONPATH=. pytest tests/.
Пример запроса к API:Bashcurl -X POST "http://localhost:8000/predict" -H "Content-Type: application/json" -d '{"LIMIT_BAL": 20000, "SEX": 2, "EDUCATION": 2, "MARRIAGE": 1, "AGE": 24, "PAY_0": 2, "BILL_AMT1": 3913, "PAY_AMT1": 0}'

## Мониторинг дрифта

Запустите скрипт: python scripts/monitor_drift.py.
Скрипт рассчитывает PSI по ключевым фичам и target, запускает контейнер и делает predict.

## CI/CD

Автоматические тесты и workflows в .github/workflows/.

Этап 1. Подготовка модели к промышленной эксплуатации
Конвертация в ONNX

Исходная модель: models/credit_default_model.pkl (sklearn Pipeline с GradientBoosting)
Конвертирована в формат ONNX с помощью skl2onnx
Скрипт: scripts/convert_to_onnx.py

## Оптимизация модели

Применена динамическая пост-тренинг квантизация (8-bit) с помощью onnxruntime
Скрипт: scripts/quantize_onnx.py

Результаты benchmark (размер модели)
Модель,Размер (MB),Уменьшение относительно .pkl
Исходная (.pkl),0.18,1.0x
ONNX,0.05,3.6x
ONNX quantized,0.05,3.6x (квантизация не дала сжатия — типично для tree-based моделей)

Вывод: Конвертация в ONNX уменьшила размер модели в 3.6 раза. Квантизация не дала дополнительного сжатия из-за природы модели (деревья решений). ONNX-версия готова к деплою в продакшен (быстрее загрузка, лучше совместимость).

## Производительность inference

Локально на CPU: предсказание одного объекта — < 1 мс (проверено в API)
Нагрузочное тестирование планируется на этапе 3 (locust/ab)

## Этап 2. Cloud Infrastructure as Code

Terraform конфигурация в infrastructure/
Создан VPC, subnets в разных зонах
Managed Kubernetes кластер с CPU node group (auto-scale 2–10)
Сервисный аккаунт с ролями (editor, k8s.admin, container-registry.images.puller)

## Этап 3. Контейнеризация и оркестрация

Multi-stage Dockerfile с ONNX моделью
FastAPI API с inference на ONNX
Kubernetes manifests в deployment/manifests:
Deployment с rolling update (3 реплики)
Service (ClusterIP)
Ingress

Локальный тест: docker run работает, Swagger на localhost:8000/docs
Деплой в Managed Kubernetes: кластер создан, manifests применены

## Этап 4. Дополненный CI/CD пайплайн

GitHub Actions workflows в .github/workflows:
build-and-test.yaml
deploy-staging.yaml
deploy-production.yaml
canary-release.yaml
rollback.yaml

Security scanning (bandit/safety)

## Этап 5. Мониторинг и observability

Configs для Prometheus/Grafana в monitoring/
Метрики приложения и инфраструктуры

## Этап 6. Мониторинг дрифта и управление моделями

PSI для data drift и concept drift в scripts/monitor_drift.py
Запуск контейнера и predict для проверки

## Этап 7. Пайплайн переобучения и автоматизация

Airflow DAG в airflow/dags/credit_scoring_retrain_dag.py (monthly retraining с триггером по дрифту)

## Документация

Архитектура MLOps пайплайна (полный цикл от данных до деплоя)
Инструкции по запуску Terraform, Docker, deploy
Локальный тест API и мониторинг дрифта