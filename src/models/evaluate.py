import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error


def evaluation(df, conformal_evaluation = True):

    # Test dataset
    df_test = df[df['split'] == 'test']

    y_test = df_test['demand_mw']
    test_pred = df_test['y_pred']
    test_errors = df_test['absolute_error']
    
    results = {
    "MAE (Mw)": [np.mean(test_errors)],
    "MAPE (%)": [np.mean(test_errors/y_test)*100],
    "RMSE (Mw)": [np.sqrt(mean_squared_error(y_test, test_pred))],
    }

    if conformal_evaluation:
        upper = df_test['upper']
        lower = df_test['lower']

        results = {
            "MAE (Mw)": [np.mean(test_errors)],
            "MAPE (%)": [np.mean(test_errors/y_test)*100],
            "RMSE (Mw)": [np.sqrt(mean_squared_error(y_test, test_pred))],
            "Coverage":[np.mean((y_test.values >= lower) & (y_test.values <= upper))],
            "Mean width (Mw)": [np.mean(upper - lower)],
            "Mean width (%)": [np.mean((upper - lower)/y_test)*100]}

    return pd.DataFrame(results)

if __name__ == "__main__":
   
    df_pred_path = Path("data/processed/predictions_2021_01.csv")
    df_pred_path_baseline = Path("data/processed/baseline_lag24h_predictions_2021_01.csv")
    df_pred_path_baseline2 = Path("data/processed/baseline_lag168h_predictions_2021_01.csv")

    df_pred = pd.read_csv(df_pred_path)
    df_pred_baseline = pd.read_csv(df_pred_path_baseline)
    df_pred_baseline2 = pd.read_csv(df_pred_path_baseline2)

    evaluation_results = evaluation(df_pred, conformal_evaluation = False)
    evaluation_results_baseline = evaluation(df_pred_baseline, conformal_evaluation = False)
    evaluation_results_baseline2 = evaluation(df_pred_baseline2, conformal_evaluation = False)

    print('XGBoost:',evaluation_results)
    print('24h lag:',evaluation_results_baseline)
    print('168h lag:',evaluation_results_baseline2)



