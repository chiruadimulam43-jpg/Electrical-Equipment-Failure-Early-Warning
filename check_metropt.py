
import pandas as pd

df = pd.read_csv("MetroPT3(AirCompressor).csv")

print("Dataset shape:", df.shape)
print("\nAll columns:")
print(df.columns.tolist())

print("\nMissing values:")
print(df.isnull().sum())

print("\nTimestamp range:")
print(df["timestamp"].min(), "to", df["timestamp"].max())

print("\nFirst 10 rows:")
print(df.head(10).to_string())
