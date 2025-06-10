'''import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dense
from keras.models import Sequential 
from keras.layers import LSTM,Dense 
def prepare_data(df):
    df = df.copy()
    df['date_time'] = pd.to_datetime(df['date_time'])
    df = df.sort_values('date_time')

    # Feature engineering
    df['hour'] = df['date_time'].dt.hour
    df['dayofweek'] = df['date_time'].dt.dayofweek
    df['month'] = df['date_time'].dt.month

    features = ['temp', 'rain_1h', 'snow_1h', 'clouds_all', 'hour', 'dayofweek', 'month']
    target = 'traffic_volume'

    return df[features], df[target]

def train_rf_model(df):
    X, y = prepare_data(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    return rf, mse

def train_lstm_model(df):
    df = df.copy()
    df['date_time'] = pd.to_datetime(df['date_time'])
    df = df.sort_values('date_time')

    series = df[['traffic_volume']].values
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(series)

    # Create sequences
    X, y = [], []
    for i in range(60, len(scaled)):
        X.append(scaled[i-60:i])
        y.append(scaled[i])
    X, y = np.array(X), np.array(y)

    model = Sequential()
    model.add(LSTM(64, return_sequences=False, input_shape=(X.shape[1], 1)))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    model.fit(X, y, epochs=5, batch_size=32, verbose=0)

    return model, scaler''' 
    
'''
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential 
from keras.layers import LSTM, Dense 

def prepare_data(df):
    df = df.copy()
    df['date_time'] = pd.to_datetime(df['date_time'])
    df = df.sort_values('date_time')

    # Feature engineering
    df['hour'] = df['date_time'].dt.hour
    df['dayofweek'] = df['date_time'].dt.dayofweek
    df['month'] = df['date_time'].dt.month

    features = ['temp', 'rain_1h', 'snow_1h', 'clouds_all', 'hour', 'dayofweek', 'month']
    target = 'traffic_volume'

    return df[features], df[target]

def train_rf_model(df):
    X, y = prepare_data(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    return rf, mse

def train_lstm_model(df):
    df = df.copy()
    df['date_time'] = pd.to_datetime(df['date_time'])
    df = df.sort_values('date_time')

    series = df[['traffic_volume']].values
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(series)

    # Create sequences
    X, y = [], []
    for i in range(60, len(scaled)):
        X.append(scaled[i-60:i])
        y.append(scaled[i])
    X, y = np.array(X), np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))  # Reshape for LSTM

    model = Sequential()
    model.add(LSTM(64, return_sequences=False, input_shape=(X.shape[1], 1)))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    model.fit(X, y, epochs=5, batch_size=32, verbose=0)

    # Optional: Evaluate
    y_pred = model.predict(X[-20:])
    mse = mean_squared_error(y[-20:], y_pred)

    return model, scaler, mse
''' 


# File: ml_models/traffic_predictor.py
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam

def prepare_data(df):
    df = df.copy()
    df['date_time'] = pd.to_datetime(df['date_time'])
    df = df.sort_values('date_time')
    df['hour'] = df['date_time'].dt.hour
    df['dayofweek'] = df['date_time'].dt.dayofweek
    df['month'] = df['date_time'].dt.month
    features = ['temp', 'rain_1h', 'snow_1h', 'clouds_all', 'hour', 'dayofweek', 'month']
    return df[features], df['traffic_volume']

def train_rf_model_with_tuning(df):
    X, y = prepare_data(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    rf = RandomForestRegressor(random_state=42)
    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [10, None],
        'min_samples_split': [2, 5]
    }
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=0)
    grid_search.fit(X_train, y_train)
    best_rf = grid_search.best_estimator_
    y_pred = best_rf.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Optimized RF RMSE: {np.sqrt(mse):.2f}")
    return best_rf, mse
from sklearn.metrics import mean_squared_error

def train_lstm_model_with_early_stopping(df):
    df = df.sort_values('date_time')
    series = df[['traffic_volume']].values
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(series)
    
    X, y = [], []
    for i in range(60, len(scaled)):
        X.append(scaled[i-60:i])
        y.append(scaled[i])
    X, y = np.array(X), np.array(y)
    
    model = Sequential([
        LSTM(64, input_shape=(X.shape[1], 1)),
        Dense(1)
    ])
    optimizer = Adam(learning_rate=0.001)
    model.compile(loss='mse', optimizer=optimizer)
    
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    model.fit(X, y, epochs=50, batch_size=32, validation_split=0.1, verbose=0, callbacks=[early_stopping])

    # Predict on training data to compute MSE (or you can use a separate test set)
    y_pred = model.predict(X, verbose=0)
    lstm_mse = mean_squared_error(y, y_pred)

    return model, scaler, lstm_mse

def evaluate_models(y_test, y_pred):
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return rmse, mae, r2

def predict_rf_in_chunks(df, chunk_size=500):
    chunks = [df[i:i+chunk_size] for i in range(0, len(df), chunk_size)]
    all_predictions = []
    for chunk in chunks:
        if len(chunk) < 60:  # skip too small
            continue
        model, _ = train_rf_model_with_tuning(chunk)
        X, _ = prepare_data(chunk)
        preds = model.predict(X)
        all_predictions.append(preds.tolist())
    return all_predictions
