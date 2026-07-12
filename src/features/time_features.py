import pandas as pd
from pathlib import Path


def features_creation():

    """
    Creation of relevant features for the demand prediction.

    inputs: 

    """
    data_processed = "data/processed/processed_data_2021_01.csv"

    df_processed = pd.read_csv(data_processed)
    df_processed['datetime'] = pd.to_datetime(df_processed['datetime'])

    # Time features
    df_processed['hour'] = df_processed['datetime'].dt.hour
    df_processed['day_of_week'] = df_processed['datetime'].dt.strftime("%W")
    df_processed['is_weekend'] =  df_processed['day_of_week'].isin([5, 6]).astype(int)
    df_processed['month'] = df_processed['datetime'].dt.month
    df_processed['day_of_year'] = df_processed['datetime'].dt.day

    # # Past features
    df_processed['demand_lag_1h'] = df_processed['demand_mw'].shift(1)
    df_processed['demand_lag_24h'] = df_processed['demand_mw'].shift(24)
    df_processed['demand_rolling_24h'] = df_processed["demand_mw"].shift(1).rolling(24).mean()

    return df_processed

if __name__ == "__main__":
    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)

    df = features_creation()
    df.to_csv(output_dir / "final_data_2021_01.csv", index=False)

    print(df[['datetime', 'hour', 'day_of_week','is_weekend', 'month', 'day_of_year',
                  'demand_lag_1h', 'demand_lag_24h', 'demand_rolling_24h']].head(5))

    print(f"Rows downloaded: {len(df)}")

