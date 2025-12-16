from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.bash import BashOperator
from datetime import datetime
import subprocess
import os

# Абсолютный путь к корню проекта 
PROJECT_ROOT = "/Users/nadezdaivanova/Documents/GitHub/credit-scoring-model"

# Функция для запуска мониторинга дрифта и возврата решения
def check_drift_and_decide(**kwargs):
    script_path = os.path.join(PROJECT_ROOT, "scripts", "monitor_drift.py")
    result = subprocess.run(["python", script_path], capture_output=True, text=True)
    print(result.stdout)
    print(result.stderr)

    # Проверяем, есть ли дрифт (по ключевым словам в выводе PSI)
    if "drift detected" in result.stdout.lower() or "PSI" in result.stdout:
        # Можно добавить парсинг PSI значения
        return 'retrain_model'
    else:
        return 'no_retrain_needed'

# Функция переобучения модели
def retrain_model():
    print("Запуск переобучения модели...")
    train_script = os.path.join(PROJECT_ROOT, "src", "train.py")  # скрипт обучения
    subprocess.run(["python", train_script], check=True)
    print("Модель переобучена и сохранена в models/new_model.onnx")

# Функция тестирования новой модели
def test_new_model():
    print("Тестирование новой модели...")
    # Пример: запуск pytest (если есть тесты)
    subprocess.run(["pytest", os.path.join(PROJECT_ROOT, "tests")], check=True)
    print("Тесты пройдены")

# Функция деплоя новой модели
def deploy_new_model():
    print("Деплой новой версии модели в K8s...")
    # Пример: обновление образа в Deployment
    subprocess.run([
        "kubectl", "set", "image", "deployment/credit-scoring-api",
        "api=cr.yandex/crp64qbt432vtdkce0rj/credit-scoring-api:new-version",
        "--kubeconfig", os.path.join(PROJECT_ROOT, "deployment", "kubeconfig.yaml")
    ], check=True)
    print("Новая версия задеплоена")

with DAG(
    dag_id='credit_scoring_retrain',
    start_date=datetime(2025, 1, 1),
    schedule_interval='@monthly',  # регулярное переобучение каждый месяц
    catchup=False,
    tags=['mlops', 'credit-scoring', 'retraining'],
    default_args={
        'owner': 'nadezhda',
        'retries': 1,
    }
) as dag:

    drift_check = BranchPythonOperator(
        task_id='check_drift',
        python_callable=check_drift_and_decide,
        provide_context=True
    )

    retrain = PythonOperator(
        task_id='retrain_model',
        python_callable=retrain_model
    )

    test = PythonOperator(
        task_id='test_new_model',
        python_callable=test_new_model
    )

    deploy = PythonOperator(
        task_id='deploy_new_model',
        python_callable=deploy_new_model
    )

    no_retrain = BashOperator(
        task_id='no_retrain_needed',
        bash_command='echo "Дрифт не обнаружен — переобучение не требуется"'
    )

    drift_check >> [retrain, no_retrain]
    retrain >> test >> deploy