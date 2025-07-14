# src/data_loader.py

import pandas as pd

def load_csv_data():
    stores = pd.read_csv("data/raw_data/stores_data_set.csv")
    features = pd.read_csv("data/raw_data/features_data_set.csv")
    sales = pd.read_csv("data/raw_data/sales_data_set.csv")
    return stores, features, sales

if __name__ == "__main__":
    stores, features, sales = load_csv_data()
    print("Data loaded successfully!")
    print(f"Stores: {stores.shape}")
    print(f"Features: {features.shape}")
    print(f"Sales: {sales.shape}")
