# Credit Scoring Model

## Описание
Проект для предсказания дефолта по кредитам с использованием машинного обучения. Реализована модель на основе GradientBoosting с метриками: AUC=0.76, precision=0.69, recall=0.31, f1=0.43.

## Структура проекта
- `src/`: Код (данные, пайплайн, обучение, API).
- `data/`: Обработанные данные (версионируются DVC).
- `tests/`: Тесты для кода.
- `models/`: Сохранённая модель.
- `mlruns/`: Данные экспериментов MLflow.
- `.github/workflows/`: Конфигурация CI/CD.
- `Dockerfile`: Контейнер для API.
- `scripts/`: Скрипты для мониторинга.

## Установка
1. Установите Docker: [docker.com](https://www.docker.com/).
2. Клонируйте репозиторий: `git clone https://github.com/nadezhdavanova896/credit-scoring-model.git`.
3. Перейдите в папку: `cd credit-scoring-model`.
4. Установите зависимости: `pip install -r requirements.txt`.

## Запуск
1. Соберите Docker-образ: `docker build -t credit-api .`.
2. Запустите контейнер: `docker run -p 8000:8000 credit-api`.
3. Проверьте API: `http://localhost:8000`.

## Тестирование
- Запустите тесты: `PYTHONPATH=. pytest tests/`.
- Пример запроса к API:
  ```bash
  curl -X POST "http://localhost:8000/predict" -H "Content-Type: application/json" -d '{"LIMIT_BAL": 20000, "SEX": 2, "EDUCATION": 2, "MARRIAGE": 1, "AGE": 24, "PAY_0": 2, "BILL_AMT1": 3913, "PAY_AMT1": 0}'
 
 ## Мониторинг
- Запустите скрипт мониторинга: `python scripts/monitor_drift.py`, который создан для расчёта PSI (Population Stability Index) и проверки API.
- Скрипт автоматически запускает и останавливает контейнер.

## CI/CD
- Автоматические тесты и DVC запускаются через GitHub Actions: [Actions](https://github.com/nadezhdavanova896/credit-scoring-model/actions).