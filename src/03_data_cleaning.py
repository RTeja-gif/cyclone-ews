import os
import pandas as pd

# Load sample data
df = pd.read_csv("data/raw/ibtracs_nio_sample.csv")

# 🔹 Convert LAT and LON columns to numeric (force invalid entries to NaN)
df["LAT"] = pd.to_numeric(df["LAT"], errors="coerce")
df["LON"] = pd.to_numeric(df["LON"], errors="coerce")

# 🔹 Drop rows where LAT or LON are missing or invalid
df = df.dropna(subset=["LAT", "LON"])

# 🔹 Filter out unrealistic coordinates
df = df[df["LAT"].between(-90, 90) & df["LON"].between(-180, 180)]

# 🔹 Save cleaned data to processed folder
os.makedirs("data/processed", exist_ok=True)
processed_data_path = "data/processed/ibtracs_nio_cleaned.csv"
df.to_csv(processed_data_path, index=False)

print(f"\n✅ Cleaned dataset saved at: {processed_data_path}")
print(f"✅ Rows after cleaning: {len(df)}")
print("✅ Data cleaning completed successfully!")
