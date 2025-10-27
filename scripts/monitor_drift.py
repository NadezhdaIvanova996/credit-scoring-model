import pandas as pd
import requests
import numpy as np
from sklearn.preprocessing import StandardScaler

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

# Загрузка тренировочных данных
train_data = pd.read_csv("data/processed/train.csv")
X_train = train_data.drop("default.payment.next.month", axis=1)
y_train_proba = pd.Series(np.random.rand(len(X_train)))  # Имитация вероятностей (замените на реальные)

# Имитация новых данных (часть тестовой выборки)
test_data = pd.read_csv("data/processed/test.csv")
X_test = test_data.drop("default.payment.next.month", axis=1)
y_test_proba = pd.Series(np.random.rand(len(X_test)))  # Имитация вероятностей

# Расчёт PSI для ключевого признака (например, LIMIT_BAL)
psi_limit_bal = calculate_psi(X_train['LIMIT_BAL'], X_test['LIMIT_BAL'])
print(f"PSI for LIMIT_BAL: {psi_limit_bal}")

# Отправка данных на API
api_url = "http://localhost:8000/predict"
sample_data = X_test.iloc[0].to_dict()
response = requests.post(api_url, json=sample_data, headers={"Content-Type": "application/json"})
if response.status_code == 200:
    print(f"API Response: {response.json()}")
else:
    print(f"API Error: {response.status_code}, {response.text}")

# Запуск API локально для теста
import subprocess
subprocess.run(["docker", "run", "-p", "8000:8000", "credit-api"], check=True)