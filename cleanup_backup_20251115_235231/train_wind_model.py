import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor

print("Loading dataset...")
df = pd.read_csv("data/processed/ibtracs_nio_cleaned.csv")
df.columns = [c.upper().strip() for c in df.columns]

# ---------------------------------------------------
# CLEAN WIND COLUMN
# ---------------------------------------------------
print("Cleaning WMO_WIND column...")

if "WMO_WIND" not in df.columns:
    raise ValueError("WMO_WIND column missing!")

# Replace blank strings with NaN
df["WMO_WIND"] = df["WMO_WIND"].replace([" ", ""], np.nan)

# Convert to numeric
df["WMO_WIND"] = pd.to_numeric(df["WMO_WIND"], errors="coerce")

# Drop rows with missing wind
df = df.dropna(subset=["WMO_WIND"])
print("Remaining rows after cleaning:", len(df))

# ---------------------------------------------------
# CLEAN LAT/LON just in case
# ---------------------------------------------------
df["LAT"] = pd.to_numeric(df["LAT"], errors="coerce")
df["LON"] = pd.to_numeric(df["LON"], errors="coerce")
df = df.dropna(subset=["LAT","LON"])

# ---------------------------------------------------
# COMPUTE TRANSLATION SPEED FEATURE
# ---------------------------------------------------
print("Computing translation speed...")

if "ISO_TIME" in df.columns:
    df["ISO_TIME"] = pd.to_datetime(df["ISO_TIME"], errors="coerce")

df = df.sort_values("ISO_TIME")

df["LAT_SHIFT"] = df["LAT"].shift(1)
df["LON_SHIFT"] = df["LON"].shift(1)
df["TIME_SHIFT"] = df["ISO_TIME"].shift(1)

def calc_speed(row):
    try:
        if pd.isna(row["LAT_SHIFT"]) or pd.isna(row["LON_SHIFT"]) or pd.isna(row["TIME_SHIFT"]):
            return np.nan
        lat1, lon1 = row["LAT_SHIFT"], row["LON_SHIFT"]
        lat2, lon2 = row["LAT"], row["LON"]
        dt = (row["ISO_TIME"] - row["TIME_SHIFT"]).total_seconds() / 3600
        if dt <= 0:
            return np.nan
        avg_lat = np.deg2rad((lat1 + lat2) / 2)
        dx = (lon2 - lon1) * 111.320 * np.cos(avg_lat)
        dy = (lat2 - lat1) * 110.574
        dist = np.sqrt(dx*dx + dy*dy)
        return dist / dt
    except:
        return np.nan

df["TRANS_SPEED"] = df.apply(calc_speed, axis=1)
df["TRANS_SPEED"] = df["TRANS_SPEED"].fillna(df["TRANS_SPEED"].median())

# ---------------------------------------------------
# TRAIN MODEL
# ---------------------------------------------------
print("Training XGBoost wind model...")

X = df[["LAT","LON","TRANS_SPEED"]]
y = df["WMO_WIND"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = XGBRegressor(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="reg:squarederror"
)

model.fit(X_train, y_train)

# ---------------------------------------------------
# EVALUATION
# ---------------------------------------------------
preds = model.predict(X_test)
mae = mean_absolute_error(y_test, preds)
r2 = r2_score(y_test, preds)

print("MAE:", mae)
print("R2:", r2)

# ---------------------------------------------------
# SAVE MODEL
# ---------------------------------------------------
os.makedirs("artifacts", exist_ok=True)
joblib.dump(model, "artifacts/wind_model.pkl")

print("\nSaved wind model -> artifacts/wind_model.pkl")
