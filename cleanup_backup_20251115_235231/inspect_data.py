import pandas as pd, sys
df = pd.read_csv("data/processed/ibtracs_nio_cleaned.csv")
df.columns = [c.upper() for c in df.columns]
print("ROWS,COLS:", df.shape)
print("COLUMNS:", df.columns.tolist())
for choice in ["SID","STORM_ID","NAME","TRACK_ID","ID"]:
    if choice in df.columns:
        group_col = choice
        break
else:
    if "SEASON" in df.columns and "NAME" in df.columns:
        df["SID"] = df["SEASON"].astype(str)+"_"+df["NAME"].astype(str)
        group_col = "SID"
    else:
        print("No storm id column found. Exiting."); sys.exit(1)
print("USING GROUP COL:", group_col)
g = df.groupby(group_col).size().sort_values(ascending=False)
print("\nTop 20 storms and counts:\n", g.head(20).to_string())
first_storm = g.index[0]
print("\nSAMPLE ROWS FOR:", first_storm)
print(df[df[group_col]==first_storm].head(20).to_string())
