import pandas as pd
from sklearn.tree import DecisionTreeRegressor
import joblib

# Columns to read to save memory
columns_to_use = [
    "tpep_pickup_datetime", "tpep_dropoff_datetime", "trip_distance",
    "passenger_count", "fare_amount", "PULocationID", "DOLocationID"
]

# Load small samples from all 3 datasets
df_jan = pd.read_parquet("yellow_tripdata_2025-01.parquet", columns=columns_to_use).head(30000)
df_feb = pd.read_parquet("yellow_tripdata_2025-02.parquet", columns=columns_to_use).head(30000)
df_mar = pd.read_parquet("yellow_tripdata_2025-03.parquet", columns=columns_to_use).head(30000)

# Combine all
df = pd.concat([df_jan, df_feb, df_mar], ignore_index=True)

# Convert datetime columns
df["tpep_pickup_datetime"] = pd.to_datetime(df["tpep_pickup_datetime"])
df["tpep_dropoff_datetime"] = pd.to_datetime(df["tpep_dropoff_datetime"])

# Feature engineering
df["hour"] = df["tpep_pickup_datetime"].dt.hour
df["dayofweek"] = df["tpep_pickup_datetime"].dt.dayofweek
df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(int)
df["is_peak_hour"] = df["hour"].isin([7, 8, 9, 17, 18, 19]).astype(int)
df["ride_duration_min"] = (df["tpep_dropoff_datetime"] - df["tpep_pickup_datetime"]).dt.total_seconds() / 60
df["dropoff_hour"] = df["tpep_dropoff_datetime"].dt.hour

# Drop missing values
df = df.dropna()

# Define final features
features = [
    "trip_distance", "passenger_count", "PULocationID", "DOLocationID",
    "hour", "dayofweek", "is_weekend", "is_peak_hour",
    "ride_duration_min", "dropoff_hour"
]

X = df[features]
y = df["fare_amount"]

# Train Decision Tree
model = DecisionTreeRegressor(random_state=42)
model.fit(X, y)

# Save model and features list
joblib.dump(model, "decision_tree_model.pkl")
joblib.dump(features, "model_features.pkl")

print("✅ Model and features saved successfully.")
