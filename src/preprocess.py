import pandas as pd
from load_data import load_data

# Columns that should NOT be used for prediction
# Reasons:
# - Identifiers (CustomerID)
# - Location noise (City, State, Zip)
# - Leakage (Churn Label, Churn Reason)

DROP_COLS = [
    "CustomerID",
    "Country",
    "State",
    "City",
    "Zip Code",
    "Lat Long",
    "Churn Label",
    "Churn Reason"
]


def preprocess_data():
    # Load raw data
    df = load_data()

    # Drop leakage and noisy columns
    df = df.drop(columns=DROP_COLS)

    # Fix Total Charges (should be numeric)
    df["Total Charges"] = pd.to_numeric(df["Total Charges"], errors="coerce")

    # Fill missing Total Charges with median
    df["Total Charges"].fillna(df["Total Charges"].median(), inplace=True)

    # Separate features and target
    X = df.drop(columns=["Churn Value"])
    y = df["Churn Value"]

    # Identify categorical and numerical columns
    categorical_cols = X.select_dtypes(include="object").columns
    numerical_cols = X.select_dtypes(exclude="object").columns

    return X, y, categorical_cols, numerical_cols


if __name__ == "__main__":
    X, y, cat_cols, num_cols = preprocess_data()

    print("Dataset shape:", X.shape)
    print("Target distribution:")
    print(y.value_counts())

    print("\nNumber of categorical columns:", len(cat_cols))
    print("Number of numerical columns:", len(num_cols))

    print("\nSample categorical columns:", list(cat_cols)[:5])
    print("Sample numerical columns:", list(num_cols)[:5])
