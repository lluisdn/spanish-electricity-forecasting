import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
import joblib

from src.models.train import data_partition
from src.models.predict import predict

def conformal_q(calibration_df, alpha=0.1, plot=True):
    q = np.quantile(calibration_df["absolute_error"], 1 - alpha)

    if plot:
        plt.hist(calibration_df["absolute_error"], bins=20, edgecolor="white", linewidth=1.2, alpha=0.85)
        plt.axvline(q, color="black", linestyle="--", label=f"Percentile {(1-alpha)*100:.0f}: {q:.2f} MW")
        plt.legend()
        plt.title("Calibration absolute errors")
        plt.xlabel("Absolute error (MW)")
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.show()

    return q


def build_conformal_predictions(calibration, test, calibration_pred, test_pred, q, alpha=0.1):
    calibration_output = calibration.copy()
    calibration_output["split"] = "calibration"
    calibration_output["y_pred"] = calibration_pred
    calibration_output["absolute_error"] = np.abs(
        calibration_output["demand_mw"] - calibration_output["y_pred"]
    )
    calibration_output["lower"] = np.nan
    calibration_output["upper"] = np.nan
    calibration_output["covered"] = np.nan

    test_output = test.copy()
    test_output["split"] = "test"
    test_output["y_pred"] = test_pred
    test_output["lower"] = test_output["y_pred"] - q
    test_output["upper"] = test_output["y_pred"] + q
    test_output["absolute_error"] = np.abs(
        test_output["demand_mw"] - test_output["y_pred"]
    )
    test_output["covered"] = (
        (test_output["demand_mw"] >= test_output["lower"])
        & (test_output["demand_mw"] <= test_output["upper"])
    )

    output_df = pd.concat([calibration_output, test_output], axis=0)

    return output_df


def plot_conformal_predictions(train, calibration, test, output_df, alpha=0.1):
    test_output = output_df[output_df["split"] == "test"]
    calibration_output = output_df[output_df["split"] == "calibration"]

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(train["datetime"], train["demand_mw"], label="Train real", color="blue")

    ax.plot(
        calibration["datetime"], calibration["demand_mw"], label="Calibration real",
        color="black",linestyle="--")

    ax.plot(calibration_output["datetime"], calibration_output["y_pred"], label="Calibration prediction",
        color="orange")

    ax.plot(test["datetime"], test["demand_mw"], label="Test real", color="black", linestyle="--")

    ax.plot(test_output["datetime"], test_output["y_pred"], label="Test prediction", color="green")

    ax.fill_between(test_output["datetime"],test_output["lower"],test_output["upper"],color="green",
        alpha=0.25,label=f"{(1-alpha)*100:.0f}% prediction interval",)

    ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d"))

    plt.xticks(rotation=45)
    plt.xlabel("January")
    plt.ylabel("Demand (MW)")
    plt.title("Spanish Electric Demand Forecast with Conformal Prediction Interval")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    _TRAIN_SIZE = 0.6
    _CAL_SIZE = 0.2
    _ALPHA = 0.1

    data_path = "data/processed/final_data_2021_01.csv"
    model_path = "models/xgboost_v1.joblib"
    output_path = Path("data/processed/predictions_2021_01.csv")

    features = ["hour","day_of_week","is_weekend","month","day_of_year",
        "demand_lag_1h","demand_lag_24h","demand_rolling_24h"]

    df = pd.read_csv(data_path)

    train, calibration, test = data_partition(df, train_size=_TRAIN_SIZE, cal_size=_CAL_SIZE)

    XGBmodel = joblib.load(model_path)

    calibration_pred = predict(XGBmodel, calibration, features)
    test_pred = predict(XGBmodel, test, features)

    cal_df = calibration.copy()
    cal_df["y_pred"] = calibration_pred
    cal_df["absolute_error"] = np.abs(cal_df["demand_mw"] - cal_df["y_pred"])

    q = conformal_q(cal_df, alpha=_ALPHA, plot=True)
    print(q)

    output_df = build_conformal_predictions(calibration=calibration, test=test, calibration_pred=calibration_pred,
        test_pred=test_pred, q=q, alpha=_ALPHA)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_path, index=False)

    print(f"Predictions saved to {output_path}")

    plot_conformal_predictions(train=train, calibration=calibration, test=test, output_df=output_df,
        alpha=_ALPHA)