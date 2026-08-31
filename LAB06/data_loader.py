import pandas as pd


def load_data(file_path):

    data = pd.read_csv(file_path)

    print("Dataset loaded successfully")
    print("Shape:", data.shape)

    print("\nFirst 5 rows:")
    print(data.head())

    return data