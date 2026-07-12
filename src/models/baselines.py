import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error 
import matplotlib.dates as mdates
import xgboost as xgb


def model():
    data = "data/processed/final_data_2021_01.csv"

    df = pd.read_csv(data)

    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.dropna().reset_index(drop=True)

    ts_cv = TimeSeriesSplit(n_splits=2)

    for train_idx, val_idx in ts_cv.split(df):
        # print(np.min(train_idx))
        # print(np.max(train_idx))

        train = df.iloc[train_idx]
        test = df.iloc[val_idx]

        features = ['hour', 'day_of_week','is_weekend', 'month', 'day_of_year',
                  'demand_lag_1h', 'demand_lag_24h', 'demand_rolling_24h']
        target = ['demand_mw']
        
        X_train = train[features]
        y_train = train[target]

        X_test = test[features]
        y_test = test[target]

        # print(X_train.max)

        xgb_reg = xgb.XGBRegressor(booster='gbtree',
                               seed=42,        
                               n_estimators=1000,
                               early_stopping_rounds=50,
                               objective='reg:squarederror',
                               reg_lambda=0.001, 
                               max_depth=5,
                               eta=0.01)

        xgb_reg.fit(X_train, y_train,
                    eval_set=[(X_train, y_train), (X_test, y_test)], verbose = False)
        
        y_pred = xgb_reg.predict(X_test)
        score = np.sqrt(mean_squared_error(y_test, y_pred))

        # print(score)

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(df['datetime'], df['demand_mw'])
        ax.plot(test['datetime'], y_pred)
        ax.plot(train['datetime'], train['demand_mw'])

        ax.xaxis.set_major_locator(mdates.DayLocator(bymonthday=None, interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%d'))

        plt.xticks(rotation=45)
        plt.xlabel('January')
        plt.ylabel('Demanda (MW)')
        plt.title('Spanish Electric Demand, January 2021')
        plt.tight_layout()
        plt.show()

    return score

final_score = model()
print(final_score)