# smart_traffic/cron.py

import pandas as pd
from .models import TrafficRecord
from django.utils.dateparse import parse_datetime

def ingest_daily_data(csv_path="data/Metro_Interstate_Traffic_Volume.csv"):
    df = pd.read_csv(csv_path)
    for _, row in df.iterrows():
        if not TrafficRecord.objects.filter(date_time=parse_datetime(row['date_time'])).exists():
            TrafficRecord.objects.create(
                date_time=parse_datetime(row['date_time']),
                holiday=row['holiday'],
                temp=row['temp'],
                rain_1h=row['rain_1h'],
                snow_1h=row['snow_1h'],
                clouds_all=row['clouds_all'],
                weather_main=row['weather_main'],
                weather_description=row['weather_description'],
                traffic_volume=row['traffic_volume']
            )
