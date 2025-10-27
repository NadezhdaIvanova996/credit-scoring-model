import pytest
import pandas as pd
from src.models.pipeline import create_pipeline
from src.data.make_dataset import load_and_split_data

def test_pipeline_creation():
    numeric_features = ['LIMIT_BAL', 'AGE']
    categorical_features = ['EDUCATION']
    pipeline = create_pipeline(numeric_features, categorical_features)
    assert pipeline is not None
    assert len(pipeline.named_steps) == 2  # preprocessor и classifier

def test_data_loading():
    data_path = "data/raw/UCI_Credit_Card.csv"
    output_dir = "data/processed/"  # Добавлен аргумент output_dir
    train, test = load_and_split_data(data_path, output_dir)  # Передаём оба аргумента
    assert len(train) > 0
    assert len(test) > 0