import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.dates as mdates
import xgboost as xgb
from pathlib import Path
import joblib
from src.config import _FEATURES, _TRAIN_SIZE, _CAL_SIZE


def data_partition(df, train_size = 0.6, cal_size = 0.2):
    if train_size + cal_size >= 1:
        raise ValueError("train_size + cal_size must be lower than 1")
    
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values("datetime")
    df = df.dropna().reset_index(drop=True)

    train_end = int(len(df) * train_size)
    cal_end = int(len(df) * (train_size + cal_size))

    train = df.iloc[:train_end]
    calibration = df.iloc[train_end:cal_end]
    test = df.iloc[cal_end:]

    return train, calibration, test


def xgbmodel(df, features):

    target = 'demand_mw'
    
    X_train = df[features]
    y_train = df[target]
    
    xgb_reg = xgb.XGBRegressor(booster='gbtree',
                               seed=42,        
                               n_estimators=1000,
                               objective='reg:squarederror',
                               reg_lambda=0.001, 
                               max_depth=5,
                               learning_rate=0.01,
                               eval_metric="rmse")

    xgb_reg.fit(X_train, y_train, verbose = False)

    return xgb_reg


if __name__ == "__main__":
    model_dir = Path("models")
    model_dir.mkdir(parents=True, exist_ok=True)

    data = "data/processed/final_data_2021_01.csv"
    df = pd.read_csv(data)

    train, calibration, test = data_partition(df, train_size = _TRAIN_SIZE, cal_size = _CAL_SIZE)

    XGBmodel = xgbmodel(train, _FEATURES)
    joblib.dump(XGBmodel, model_dir / "xgboost_v1.joblib")


