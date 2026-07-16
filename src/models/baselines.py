import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.dates as mdates
import xgboost as xgb
from pathlib import Path


def xgbmodel(alpha=0.1, train_size=0.6, cal_size = 0.2):
    data = "data/processed/final_data_2021_01.csv"

    df = pd.read_csv(data)

    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.dropna().reset_index(drop=True)

    n = len(df)

    train_end = int(n * train_size)
    cal_end = int(n * (train_size + cal_size))

    train = df.iloc[:train_end]
    calibration = df.iloc[train_end:cal_end]
    test = df.iloc[cal_end:]

    features = ['hour', 'day_of_week','is_weekend', 'month', 'day_of_year',
            'demand_lag_1h', 'demand_lag_24h', 'demand_rolling_24h']
    target = 'demand_mw'
    
    X_train = train[features]
    y_train = train[target]
    
    X_cal = calibration[features]
    y_cal = calibration[target]

    X_test = test[features]
    y_test = test[target]

    xgb_reg = xgb.XGBRegressor(booster='gbtree',
                               seed=42,        
                               n_estimators=1000,
                               objective='reg:squarederror',
                               reg_lambda=0.001, 
                               max_depth=5,
                               eta=0.01,
                               eval_metric="rmse")

    xgb_reg.fit(X_train, y_train, verbose = False)

    cal_pred = xgb_reg.predict(X_cal)

    cal_errors = np.abs(y_cal - cal_pred)

    cal_cp_df = calibration.copy()
    cal_cp_df['prediction'] = cal_pred
    cal_cp_df['errors'] = cal_errors

    def conformal_prediction(calibration_df, plot=True, alpha=alpha):
        q = np.quantile(calibration_df['errors'], 1 - alpha)

        if plot:
            plt.hist(calibration_df['errors'], bins = 20, edgecolor='white', linewidth=1.2, alpha=0.85)
            plt.axvline(q, color='black', linestyle = '--', label = f'Percentil 90, {q:.2f}')
            plt.show()

        return q

    q = conformal_prediction(cal_cp_df)
        
    test_pred = xgb_reg.predict(X_test)
    lower = test_pred - q
    upper = test_pred + q
    test_errors = np.abs(y_test - test_pred)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(train['datetime'], train['demand_mw'], label= 'Train', color = 'blue')
    ax.plot(calibration['datetime'], cal_pred, label = 'Calibration (preds)',  color = 'orange')
    ax.plot(calibration['datetime'], calibration['demand_mw'], label = 'Calibration (real)', color = 'black', linestyle = '--')
    ax.plot(test['datetime'], test_pred, label = 'Test (preds)', color = 'green')
    ax.fill_between(test['datetime'], lower, upper, color='green', 
                    alpha=0.5, label=f"{(1-alpha)*100}% prediction interval")
    ax.plot(test['datetime'], y_test, label = 'Test (real)', color = 'black', linestyle = '--')

    ax.xaxis.set_major_locator(mdates.DayLocator(bymonthday=None, interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d'))

    plt.xticks(rotation=45)
    plt.xlabel('January')
    plt.ylabel('Demand (MW)')
    plt.title('Spanish Electric Demand, January 2021')
    plt.tight_layout()
    plt.legend()
    plt.show()

    output_df = df.copy()

    output_df.loc[train.index, "split"] = "train"
    output_df.loc[calibration.index, "split"] = "calibration"
    output_df.loc[test.index, "split"] = "test"

    output_df["y_pred"] = np.nan
    output_df["lower"] = np.nan
    output_df["upper"] = np.nan
    output_df["absolute_error"] = np.nan

    output_df.loc[calibration.index, "y_pred"] = cal_pred
    output_df.loc[calibration.index, "absolute_error"] = cal_errors

    output_df.loc[test.index, "y_pred"] = test_pred
    output_df.loc[test.index, "absolute_error"] = test_errors
    output_df.loc[test.index, "lower"] = lower
    output_df.loc[test.index, "upper"] = upper

    return output_df


if __name__ == "__main__":
    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_df = xgbmodel()
    output_df.to_csv(output_dir / "predictions_2021_01.csv", index=False)

