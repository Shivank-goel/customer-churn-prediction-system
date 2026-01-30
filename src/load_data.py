# Data loading utilities for the customer churn prediction project

import pandas as pd

def load_data (path="../data/raw.csv"):
    df = pd.read_csv(path)
    return df

if __name__ == "__main__":
    df = load_data()
    print(df.head())
    print(df.shape)