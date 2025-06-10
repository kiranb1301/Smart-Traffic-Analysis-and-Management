import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from ml_models.data_loader import load_static_data
from math import sqrt

def optimize_traffic(model_type='rf', intersections=["A", "B", "C"]):
    df = load_static_data()
    if df.empty:
        return {}, "No data loaded"

    features = ["temp", "rain_1h", "snow_1h", "clouds_all", "hour", "dayofweek", "month", "year"]
    X = df[features]
    y = df["traffic_volume"]

    if model_type == 'rf':
        model = RandomForestRegressor()
    else:
        model = DecisionTreeRegressor()

    model.fit(X, y)
    predictions = model.predict(X[:len(intersections)])
    optimized = {inter: int(pred) // 100 for inter, pred in zip(intersections, predictions)}

    rmse = sqrt(mean_squared_error(y, model.predict(X)))
    if isinstance(model_type, dict):
        model_type = 'unknown'
    return optimized, f"{model_type.upper()} RMSE: {rmse:.2f}"

def optimize_traffic_in_chunks(df, chunk_size=500):
    results = []
    for i in range(0, len(df), chunk_size):
        chunk = df.iloc[i:i+chunk_size]
        if len(chunk) < 10:
            continue
        optimized, rmse = optimize_traffic(model_type="rf", intersections=["A", "B", "C"], data=chunk)
        results.append({"chunk_index": i//chunk_size, "optimized": optimized, "rmse": rmse})
    return results
