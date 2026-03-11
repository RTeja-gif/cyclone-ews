import pandas as pd
df = pd.read_csv("data/processed/ibtracs_nio_cleaned.csv")
cols = ["WMO_PRES","PRESSURE","P","WMO_WIND","WIND","WIND_SPEED","MAX_WIND"]
found = [c for c in cols if c in df.columns]
print("Found candidate columns:", found)
for c in found:
    print(c, "non-null count:", df[c].notna().sum(), " total:", len(df))
# show a small sample of the first 10 rows for these columns
print("\nSample rows (first 10) for found columns:")
print(df[found].head(10).to_string())
