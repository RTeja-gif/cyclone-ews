import pandas as pd

df = pd.read_csv("data/processed/ibtracs_nio_cleaned.csv")
print("Shape:", df.shape)
print("Columns:", df.columns.tolist())
print("Preview:\n", df.head().to_string())
