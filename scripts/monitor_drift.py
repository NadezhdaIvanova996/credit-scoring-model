import pandas as pd
import requests
import numpy as np
from sklearn.preprocessing import StandardScaler
import subprocess
import time

def calculate_psi(expected, actual, bucket=10, axis=0):
    """Calculate Population Stability Index (PSI)."""
    def get_bins(a, buckets):
        limits = np.linspace(min(a), max(a), buckets + 1)
        return np.digitize(a, limits)
    expected_binned = get_bins(expected, bucket)
    actual_binned = get_bins(actual, bucket)
    total_expected = len(expected)
    total_actual = len(actual)
    psi = 0
    for i in range(bucket + 1):
        exp_count = np.sum(expected_binned == i)
        act_count = np.sum(actual_binned == i)
        exp_pct = exp_count / total_expected
        act_pct = act_count / total_actual
        if exp_pct == 0:
            exp_pct = 0.0001
        if act_pct == 0:
            act_pct = 0.0001
        psi += (act_pct - exp_pct) * np.log(act_pct / exp_pct)
    return psi

# Функция для запуска контейнера
def start_container():
    try:
        # Остановка существующих контейнеров credit-api
        subprocess.run(["docker", "stop", "credit-api-monitor"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
        subprocess.run(["docker", "rm", "credit-api-monitor"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
        # Запуск нового контейнера
        subprocess.run(["docker", "run", "-d", "-p", "8000:8000", "--name", "credit-api-monitor", "credit-api"], check=True)
        time.sleep(5)  # Дать время на запуск
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error starting container: {e}")
        return False

# Загрузка тренировочных данных
train_data = pd.read_csv("data/processed/train.csv")
X_train = train_data.drop("default.payment.next.month", axis=1)
y_train_proba = pd.Series(np.random.rand(len(X_train)))  # Имитация вероятностей

# Имитация новых данных
test_data = pd.read_csv("data/processed/test.csv")
X_test = test_data.drop("default.payment.next.month", axis=1)
y_test_proba = pd.Series(np.random.rand(len(X_test)))  # Имитация вероятностей

# Расчёт PSI
psi_limit_bal = calculate_psi(X_train['LIMIT_BAL'], X_test['LIMIT_BAL'])
print(f"PSI for LIMIT_BAL: {psi_limit_bal}")

# Запуск контейнера
if start_container():
    api_url = "http://localhost:8000/predict"
    sample_data = X_test.iloc[0].to_dict()
    response = requests.post(api_url, json=sample_data, headers={"Content-Type": "application/json"})
    if response.status_code == 200:
        print(f"API Response: {response.json()}")
    else:
        print(f"API Error: {response.status_code}, {response.text}")
else:
    print("Failed to start container, using existing API if available.")
    response = requests.post("http://localhost:8000/predict", json=sample_data, headers={"Content-Type": "application/json"})
    if response.status_code == 200:
        print(f"API Response: {response.json()}")
    else:
        print(f"API Error: {response.status_code}, {response.text}")

# Остановка контейнера
subprocess.run(["docker", "stop", "credit-api-monitor"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
subprocess.run(["docker", "rm", "credit-api-monitor"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)