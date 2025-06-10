import pandas as pd
from pathlib import Path
from django.utils import timezone

DATA_PATH = Path("data/Metro_Interstate_Traffic_Volume.csv")

def load_static_data(file_path=DATA_PATH):
    try:
        df = pd.read_csv(file_path)

        # Convert 'date_time' to datetime, coerce invalids
        df['date_time'] = pd.to_datetime(df['date_time'], errors='coerce')

        # Drop rows with invalid timestamps
        df.dropna(subset=['date_time'], inplace=True)

        # Convert naive datetime to aware datetime
        df['date_time'] = df['date_time'].apply(timezone.make_aware)

        # Fill missing numeric values with mean
        numeric_cols = df.select_dtypes(include='number').columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

        # Feature Engineering
        df['hour'] = df['date_time'].dt.hour
        df['dayofweek'] = df['date_time'].dt.dayofweek
        df['month'] = df['date_time'].dt.month
        df['year'] = df['date_time'].dt.year

        return df

    except Exception as e:
        print(f"[ERROR] Failed to load or process data: {e}")
        return pd.DataFrame()  # Return empty DataFrame on failure
