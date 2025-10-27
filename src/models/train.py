import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, roc_curve
import matplotlib.pyplot as plt
import pandas as pd
import os
import joblib  # Добавлен импорт joblib

# Загрузка данных
data = pd.read_csv("/Users/nadezdaivanova/Documents/GitHub/credit-scoring-model/data/processed/train.csv")
X = data.drop("default.payment.next.month", axis=1)
y = data["default.payment.next.month"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Определение признаков
numeric_features = ['LIMIT_BAL', 'AGE', 'BILL_AMT1', 'PAY_AMT1']
categorical_features = ['EDUCATION', 'MARRIAGE', 'PAY_0']

with mlflow.start_run():
    # Создание и обучение пайплайна
    from pipeline import create_pipeline
    pipeline = create_pipeline(numeric_features, categorical_features)
    pipeline.fit(X_train, y_train)

    # Предсказания и метрики
    y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
    y_pred = pipeline.predict(X_test)
    auc = roc_auc_score(y_test, y_pred_proba)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Логирование в MLflow
    mlflow.log_param("model_type", "GradientBoosting")
    mlflow.log_params(pipeline['classifier'].get_params())
    mlflow.log_metrics({"test_auc": auc, "test_precision": precision, "test_recall": recall, "test_f1": f1})

    # ROC-кривая
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.savefig("roc_curve.png")
    mlflow.log_artifact("roc_curve.png")

    # Сохранение модели
    mlflow.sklearn.log_model(pipeline, "model")
    import joblib
    model_path = "models/credit_default_model.pkl"
    os.makedirs(os.path.dirname(model_path), exist_ok=True)  # Создаём папку, если её нет
    joblib.dump(pipeline, model_path)  # Сохраняем модель в файл
    mlflow.log_artifact(model_path)  # Логируем файл в MLflow

    # Сохранение метрик
    with open("metrics.json", "w") as f:
        f.write(f'{{"auc": {auc}, "f1": {f1}}}')