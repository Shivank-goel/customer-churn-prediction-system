# Exploratory Data Analysis module

from load_data import load_data

df = load_data()

print(df.info())
print(df.isnull().sum())
print(df["Churn Label"].value_counts())