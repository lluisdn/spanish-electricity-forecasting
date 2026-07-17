from pathlib import Path
import pandas as pd
import joblib
from src.models.baselines import Lag24hBaseline
from src.models.train import data_partition
from src.config import _FEATURES

def predict(model, df, features):
    predictions = model.predict(df[features])
    return predictions

if __name__ == "__main__":

    data = "data/processed/final_data_2021_01.csv"
    df = pd.read_csv(data)

    train, calibration, test = data_partition(df)

    XGBmodel = joblib.load("models/xgboost_v1.joblib")

    train_pred = predict(XGBmodel, train, _FEATURES)
