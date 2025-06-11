import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Input, LSTM, Dense
import joblib
import os

# -----------------------------
# 1. Load and preprocess data
# -----------------------------
import pandas as pd
from pathlib import Path
def load_and_split_chunks(chunk_size=500):
    df = pd.read_csv(Path("data/Synthetic_Traffic_Volume_Data.csv"))
    print(f"📊 Loaded rows: {len(df)}")

    df = df.fillna(0)
    chunks = [df[i:i+chunk_size] for i in range(0, len(df), chunk_size)]
    print(f"📦 Total chunks created: {len(chunks)}")
    return chunks


# -----------------------------
# 2. Train Random Forest once
# -----------------------------
def train_rf_once(df):
    X = df[["temp", "rain_1h", "snow_1h"]].fillna(0)
    y = df["traffic_volume"]
    model = RandomForestRegressor(n_estimators=100)
    model.fit(X, y)
    return model

# -----------------------------
# 3. Train LSTM once
# -----------------------------
def train_lstm_once(df):
    data = df[["traffic_volume"]].fillna(0).values
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)

    X, y = [], []
    for i in range(10, len(scaled_data)):
        X.append(scaled_data[i-10:i])
        y.append(scaled_data[i])
    X, y = np.array(X), np.array(y)

    model = Sequential()
    model.add(Input(shape=(X.shape[1], 1)))
    model.add(LSTM(64))
    model.add(Dense(1))
    model.compile(optimizer="adam", loss="mse")
    model.fit(X, y, epochs=3, batch_size=32, verbose=0)

    return model, scaler

# -----------------------------
# 4. Clustering Logic
# -----------------------------
def run_clustering(chunk):
    features = chunk[["temp", "rain_1h", "snow_1h"]].fillna(0)
    kmeans = KMeans(n_clusters=3, n_init=10)
    chunk.loc[:, 'cluster'] = kmeans.fit_predict(features)
    cluster_summary = chunk.groupby("cluster")["traffic_volume"].mean().to_dict()
    return cluster_summary

# -----------------------------
# 5. Optimization (mocked)
# -----------------------------
def optimize_lights(chunk):
    avg_volume = chunk["traffic_volume"].mean()
    return {
        "A": int(30 + avg_volume % 20),
        "B": int(25 + avg_volume % 15),
        "C": int(20 + avg_volume % 10),
    }

# -----------------------------
# 6. Main Realtime Simulator
# -----------------------------
def simulate_realtime_from_chunks(chunk_size=500):
    chunks = load_and_split_chunks(chunk_size)
    
    # Train ML models on full data
    full_df = pd.concat(chunks)
    rf_model = train_rf_once(full_df)
    lstm_model, lstm_scaler = train_lstm_once(full_df)

    results = []
    for idx, chunk in enumerate(chunks):
        log = f"✅ Processing Chunk {idx+1}/{len(chunks)}"

        # RF Prediction
        rf_X = chunk[["temp", "rain_1h", "snow_1h"]].fillna(0)
        rf_preds = rf_model.predict(rf_X)
        rf_avg = np.mean(rf_preds)

        # LSTM Prediction
        recent_data = chunk["traffic_volume"].fillna(0).values[-10:]
        if len(recent_data) < 10:
            print(f"⚠️ Skipping Chunk {idx+1} (not enough data for LSTM)")
            continue

        scaled_input = lstm_scaler.transform(recent_data.reshape(-1, 1))
        X_lstm = np.reshape(scaled_input, (1, 10, 1))
        lstm_pred = lstm_model.predict(X_lstm)[0][0]
        lstm_pred_actual = lstm_scaler.inverse_transform([[lstm_pred]])[0][0]

        # Clustering and Optimization
        clusters = run_clustering(chunk)
        optimized = optimize_lights(chunk)

        result = {
            "chunk_id": idx + 1,
            "rf_avg_pred": float(rf_avg),
            "lstm_pred": float(lstm_pred_actual),
            "cluster_volume_avg": clusters,
            "optimized_lights": optimized,
            "log": log
        }

        print(log)
        results.append(result)

    print("✅ Simulation complete.")
    print(f"🧠 Total Chunks Processed: {len(results)}")
    return results 