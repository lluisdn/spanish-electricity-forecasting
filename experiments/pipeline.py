from pathlib import Path
import joblib
import pandas as pd
from src.config import _TRAIN_SIZE, _CAL_SIZE
from src.data.ingest_electricity import download_ree_demand
from src.data.ingest_weather import download_open_meteo_weather
from src.data.build_dataset import build_dataset
from src.features.time_features import features_creation
from src.models.train import data_partition, xgbmodel
from src.models.baselines import build_baseline_output
from src.models.predict import predict
from src.models.evaluate import evaluation

def pipeline(period_name, start_date="2021-01-01T00:00", end_date="2021-12-31T23:59"):
    # DATA 
    output_dir = Path("data/raw")
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir = Path("models")
    model_dir.mkdir(parents=True, exist_ok=True)

    electric_demand = download_ree_demand(start_date, end_date)
    # electric_demand.to_csv(output_dir / f"ree_demand_{period_name}.csv", index=False)

    start_date_adapt = start_date.split("T")[0]
    end_date_adapt = end_date.split("T")[0]

    madrid_weather = download_open_meteo_weather(latitude=40.4168, longitude=-3.7038,
        start_date=start_date_adapt,end_date=end_date_adapt, location_name="madrid")
    # madrid_weather.to_csv(output_dir / f"weather_madrid_{period_name}.csv", index=False)

    df = build_dataset(electric_demand, madrid_weather)
    # df.to_csv(output_dir / f"processed_data_{period_name}.csv", index=False)

    # FEATURES
    df_processed = features_creation(df)

    features =["hour", "day_of_week", "is_weekend", "month", "day_of_year","demand_lag_1h",
    "demand_lag_24h", "demand_rolling_24h"]

    #TRAIN
    train, calibration, test = data_partition(df_processed, train_size = _TRAIN_SIZE, 
        cal_size = _CAL_SIZE)

    XGBmodel = xgbmodel(train, features)
    # joblib.dump(XGBmodel, model_dir / f"xgboost_{period_name}_{}.joblib")

    #PREDICT
    train_pred = predict(train, XGBmodel, features)
    calibration_pred = predict(calibration, XGBmodel, features)
    test_pred = predict(test, XGBmodel, features)
    
    #EVALUATION

    train_output = build_baseline_output(df=train, predictions=train_pred, split_name="train")
    calibration_output = build_baseline_output(df=calibration, predictions=calibration_pred, 
        split_name="calibration")
    test_output = build_baseline_output(df=test, predictions=test_pred, split_name="test")

    evaluation_df = pd.concat([train_output, calibration_output, test_output], axis=0)

    evaluation_results = evaluation(evaluation_df, conformal_evaluation = False)

    print(evaluation_results)

    return 0

if __name__ == "__main__":
    pipeline(2021)

