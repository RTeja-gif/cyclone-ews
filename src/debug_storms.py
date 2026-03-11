import pandas as pd
import numpy as np
from datetime import datetime

df = pd.read_csv("data/processed/ibtracs_nio_cleaned.csv")
df.columns = [c.upper().strip() for c in df.columns]

print("TOTAL ROWS:", len(df))
# show sample header and first rows
print("\nCOLUMNS (sample):", df.columns.tolist()[:50])
print("\nFIRST 5 ROWS (LAT,LON,SEASON,ISO_TIME,SID):")
print(df.loc[:,["LAT","LON","SEASON","ISO_TIME","SID"]].head(5).to_string())

# SEASON summary
if "SEASON" in df.columns:
    seasons = pd.to_numeric(df["SEASON"], errors="coerce")
    print("\nSEASON unique count:", seasons.nunique(dropna=True))
    print("SEASON sample values (sorted unique, first 20):", sorted([int(x) for x in seasons.dropna().unique()])[:20])
    # count rows in 2000-2025
    in_range = df[(seasons >= 2000) & (seasons <= 2025)]
    print("Rows with SEASON in 2000-2025:", len(in_range))
else:
    print("\nNo SEASON column present")

# ISO_TIME summary
if "ISO_TIME" in df.columns:
    df["ISO_TIME_parsed"] = pd.to_datetime(df["ISO_TIME"], errors="coerce")
    print("\nISO_TIME - parsed non-null:", df["ISO_TIME_parsed"].notna().sum(), " / ", len(df))
    if df["ISO_TIME_parsed"].notna().any():
        top = df.groupby("SID")["ISO_TIME_parsed"].max().sort_values(ascending=False).head(30)
        print("\nTop 30 groups by latest ISO_TIME year:")
        print(top.apply(lambda x: x.strftime('%Y-%m-%d %H:%M') if not pd.isna(x) else 'NA').to_string())
else:
    print("\nISO_TIME column is not present or wrongly named")

# show top 40 SIDs by count and their sample SEASON / last ISO_TIME
groups = df.groupby("SID").agg(count=("SID","size"), season_first=("SEASON", lambda s: s.iloc[0] if len(s)>0 else None), last_time=("ISO_TIME", lambda s: s.dropna().max() if len(s.dropna())>0 else None)).sort_values("count", ascending=False).head(40)
print("\nTop 40 SIDs with counts, sample season, last ISO_TIME:")
print(groups.to_string())

# small sample of SIDs that would be selected by the 2000-2025 filter
candidates=[]
for sid, g in df.groupby("SID"):
    try:
        s = int(pd.to_numeric(g["SEASON"].iloc[0], errors="coerce"))
    except:
        s = None
    if s and 2000 <= s <= 2025:
        candidates.append(sid)
print("\nNumber of SIDs with SEASON in 2000-2025 (unique):", len(set(candidates)))
print("Sample 10 of those SIDs:", list(set(candidates))[:10])
