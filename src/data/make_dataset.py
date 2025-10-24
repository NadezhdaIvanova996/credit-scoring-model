import sys
import pandas as pd
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    input_path = sys.argv[1]
    output_dir = sys.argv[2]
    data = pd.read_csv(input_path)
    data = data.drop("ID", axis=1, errors="ignore")
    data['AGE_BIN'] = pd.cut(data['AGE'], bins=[18, 30, 40, 50, 100], labels=[0, 1, 2, 3])
    train, test = train_test_split(data, test_size=0.2, random_state=42)
    train.to_csv(f"{output_dir}/train.csv", index=False)
    test.to_csv(f"{output_dir}/test.csv", index=False)