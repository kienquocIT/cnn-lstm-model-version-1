import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def preprocess_data(df, scaler_car_count=None, scaler_hour=None):
    df = df.resample('5min').mean().interpolate()
    df['car_count'] = df['car_count'].round().astype(int)
    df = df.ffill().bfill()
    df['hour'] = df.index.hour

    if scaler_car_count is None:
        scaler_car_count = MinMaxScaler()
        df['car_count'] = scaler_car_count.fit_transform(df[['car_count']])
    else:
        df['car_count'] = scaler_car_count.transform(df[['car_count']])

    if scaler_hour is None:
        scaler_hour = MinMaxScaler()
        df['hour'] = scaler_hour.fit_transform(df[['hour']])
    else:
        df['hour'] = scaler_hour.transform(df[['hour']])

    return df, scaler_car_count, scaler_hour
