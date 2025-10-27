import sys
import pandas as pd
from sklearn.model_selection import train_test_split

def load_and_split_data(data_path, output_dir):
    data = pd.read_csv(data_path)
    data = data.drop("ID", axis=1, errors="ignore")
    data['AGE_BIN'] = pd.cut(data['AGE'], bins=[18, 30, 40, 50, 100], labels=[0, 1, 2, 3])
    train, test = train_test_split(data, test_size=0.2, random_state=42)
    train.to_csv(f"{output_dir}/train.csv", index=False)
    test.to_csv(f"{output_dir}/test.csv", index=False)
    return train, test  # Возвращаем данные для тестов (опционально)

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python make_dataset.py <input_path> <output_dir>")
        sys.exit(1)
    input_path = sys.argv[1]
    output_dir = sys.argv[2]
    load_and_split_data(input_path, output_dir)