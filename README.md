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

## Этап 1. Подготовка модели к промышленной эксплуатации

### Конвертация в ONNX
- Исходная модель: `models/credit_default_model.pkl` (sklearn Pipeline с GradientBoosting)
- Конвертирована в формат ONNX с помощью `skl2onnx`
- Скрипт: `scripts/convert_to_onnx.py`

### Оптимизация модели
- Применена динамическая пост-тренинг квантизация (8-bit) с помощью `onnxruntime`
- Скрипт: `scripts/quantize_onnx.py`

### Результаты benchmark (размер модели)

| Модель                  | Размер (MB) | Уменьшение относительно .pkl |
|-------------------------|-------------|-----------------------------|
| Исходная (.pkl)         | 0.18       | 1.0x                        |
| ONNX                    | 0.05       | **3.6x**                    |
| ONNX quantized          | 0.05       | 3.6x (квантизация не дала сжатия — типично для tree-based моделей) |

**Вывод**: Конвертация в ONNX уменьшила размер модели в 3.6 раза. Квантизация не дала дополнительного сжатия из-за природы модели (деревья решений). ONNX-версия готова к деплою в продакшен (быстрее загрузка, лучше совместимость).

### Производительность inference
- Локально на CPU: предсказание одного объекта — < 1 мс (проверено в API)
- Нагрузочное тестирование планируется на этапе 3 (locust/ab)