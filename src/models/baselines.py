import pandas as pd
from pathlib import Path

from src.models.train import data_partition
from src.config import _TRAIN_SIZE, _CAL_SIZE

class Lag24hBaseline:

    def fit(self, X, y=None):
        return self

    def predict(self, X):
        return X["demand_lag_24h"].values


def build_baseline_output(df, predictions, split_name):

    output_df = df.copy()
    output_df["split"] = split_name
    output_df["model"] = "lag_24h_baseline"
    output_df["y_pred"] = predictions
    output_df["absolute_error"] = (
        output_df["demand_mw"] - output_df["y_pred"]
    ).abs()

    return output_df


class Lag168hBaseline:

    def fit(self, X, y=None):
        return self

    def predict(self, X):
        return X["demand_lag_168h"].values


def build_baseline_output(df, predictions, split_name):

    output_df = df.copy()
    output_df["split"] = split_name
    output_df["model"] = "lag_24h_baseline"
    output_df["y_pred"] = predictions
    output_df["absolute_error"] = (
        output_df["demand_mw"] - output_df["y_pred"]
    ).abs()

    return output_df


if __name__ == "__main__":
    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)

    data_path = "data/processed/final_data_2021_01.csv"
    output_path = output_dir / "baseline_lag24h_predictions_2021_01.csv"
    output_path2 = output_dir / "baseline_lag168h_predictions_2021_01.csv"

    df = pd.read_csv(data_path)

    train, calibration, test = data_partition(df, train_size=_TRAIN_SIZE, cal_size=_CAL_SIZE)

    baseline_model = Lag24hBaseline()
    baseline_model.fit(train, train["demand_mw"])

    baseline_model2 = Lag168hBaseline()
    baseline_model2.fit(train, train["demand_mw"])

    train_pred = baseline_model.predict(train)
    calibration_pred = baseline_model.predict(calibration)
    test_pred = baseline_model.predict(test)

    train_pred2 = baseline_model2.predict(train)
    calibration_pred2 = baseline_model2.predict(calibration)
    test_pred2 = baseline_model2.predict(test)

    train_output = build_baseline_output(df=train, predictions=train_pred, split_name="train")
    calibration_output = build_baseline_output(df=calibration, predictions=calibration_pred,
        split_name="calibration")
    test_output = build_baseline_output(df=test, predictions=test_pred, split_name="test")
    
    train_output2 = build_baseline_output(df=train, predictions=train_pred2, split_name="train")
    calibration_output2 = build_baseline_output(df=calibration, predictions=calibration_pred2,
        split_name="calibration")
    test_output2 = build_baseline_output(df=test, predictions=test_pred2, split_name="test")

    output_df = pd.concat([train_output, calibration_output, test_output], axis=0)
    output_df.to_csv(output_path, index=False)

    output_df2 = pd.concat([train_output2, calibration_output2, test_output2], axis=0)
    output_df2.to_csv(output_path2, index=False)

    print(f"Baseline predictions saved to {output_path}")
    print(f"Baseline predictions saved to {output_path2}")
    # print(output_df.head())