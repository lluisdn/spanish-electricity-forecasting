from pathlib import Path
import pandas as pd
import joblib
from src.models.train import data_partition


def predict(model, df, features):
    predictions = model.predict(df[features])
    return predictions

if __name__ == "__main__":

    _TRAIN_SIZE = 0.6
    _CAL_SIZE = 0.2

    data = "data/processed/final_data_2021_01.csv"
    df = pd.read_csv(data)

    train, calibration, test = data_partition(df)

    XGBmodel = joblib.load("models/xgboost_v1.joblib")
    features = ['hour', 'day_of_week','is_weekend', 'month', 'day_of_year',
            'demand_lag_1h', 'demand_lag_24h', 'demand_rolling_24h']

    train_pred = predict(XGBmodel, train, features)
