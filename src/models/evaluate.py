import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error 


def evaluation(alpha = 0.1):
    data = "data/processed/predictions_2021_01.csv"

    df = pd.read_csv(data)

    # Calibration dataset
    df_cal = df[df['split'] == 'calibration']

    q = np.quantile(df_cal['absolute_error'], 1 - alpha)

    # Test dataset
    df_test = df[df['split'] == 'test']

    y_test = df_test['demand_mw']
    test_pred = df_test['y_pred']
    test_errors = df_test['absolute_error']
    upper = df_test['upper']
    lower = df_test['lower']

    coverage = np.mean((y_test.values >= lower) & (y_test.values <= upper))

    plt.hist(test_errors, bins = 20, edgecolor='white', linewidth=1.2, alpha=0.85)
    plt.axvline(q, color='black', linestyle = '--', label = f'Percentil 90, {q:.2f}')
    plt.show()

    print('Coverage:', coverage, 'vs', 1 - alpha)

    print('Mean width (Mw):', np.mean(upper - lower))
    print('Mean width (%):', np.mean((upper - lower)/y_test)*100)
    print('Mean abs. error (Mw):', np.mean(test_errors))
    print('Mean squared error:', np.mean( np.sqrt(mean_squared_error(y_test, test_pred))))

    return 0

if __name__ == "__main__":
    evaluation()    


