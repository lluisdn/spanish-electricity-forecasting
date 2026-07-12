import pandas as pd
from pathlib import Path

def build_dataset(verbose = False):
   
    data_electricity = "data/raw/ree_demand_2021_01.csv"
    data_weather = "data/raw/weather_madrid_2021_01.csv"

    df_electricity = pd.read_csv(data_electricity)
    df_weather = pd.read_csv(data_weather)

    #Cheks:
    if df_electricity.shape[0] != df_weather.shape[0]:
        if verbose:
            print('Datasets have different size')

        n_dupl_elec = df_electricity.duplicated().sum()  
        n_dupl_weather = df_weather.duplicated().sum()  
        
        if n_dupl_elec >= 1 or n_dupl_weather >= 1:
            if n_dupl_elec  >= 1:
                df_electricity = df_electricity.drop_duplicates()
            elif n_dupl_weather  >= 1:
                df_weather = df_weather.drop_duplicates()
            if verbose:
                print('Deleting duplicates')

    df_electricity['datetime'] = pd.to_datetime(df_electricity['timestamp'], unit='s')
    df_weather['datetime'] = pd.to_datetime(df_weather['timestamp'], unit='s')
    df_electricity["datetime"] = df_electricity["datetime"].dt.tz_localize(None) 

    if verbose:
        print('Left join...')

    df = pd.merge(df_electricity[['datetime','demand_mw']], df_weather, on='datetime', how='inner')

    # Dataset validation
    print('df shape:', df.shape)
    print('df missings:', df.isna().sum())
    expected_range = pd.date_range(
        start=df["datetime"].min(),
        end=df["datetime"].max(),
        freq="h")
    missing_datetimes = expected_range.difference(df["datetime"])
    print(f"Missing hours: {len(missing_datetimes)}")

    return df

if __name__ == "__main__":
    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)

    df = build_dataset(verbose=True)

    df.to_csv(output_dir / "processed_data_2021_01.csv", index=False)

    print(df.head())
    print(f"Rows downloaded: {len(df)}")