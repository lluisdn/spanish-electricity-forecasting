import requests
import pandas as pd
from pathlib import Path

 # Caso fácil
 
def download_ree_demand(start_date: str, end_date: str, time_trunc = "hour" ) -> pd.DataFrame:
    """
    Download electricity demand data from REData API.

    Parameters
    ----------
    start_date : str
        Start datetime in format YYYY-MM-DDTHH:MM
    end_date : str
        End datetime in format YYYY-MM-DDTHH:MM
    time_trunc : str
        hour / day / month / year

    Returns
    -------
    pd.DataFrame
        DataFrame with timestamp and demand_mw.
    """

    url = "https://apidatos.ree.es/es/datos/demanda/evolucion"

    params = {
        "start_date": start_date,
        "end_date": end_date,
        "time_trunc": time_trunc,
        "geo_trunc": "electric_system",
        "geo_limit": "peninsular",
        "geo_ids": "8741",
    }

    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()

    data = response.json()

    values = data["included"][0]["attributes"]["values"]

    df = pd.DataFrame(values)
    df = df.rename(columns={"datetime": "timestamp", "value": "demand_mw"})
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df[["timestamp", "demand_mw"]]

    return df


if __name__ == "__main__":
    output_dir = Path("data/raw")
    output_dir.mkdir(parents=True, exist_ok=True)

    df = download_ree_demand(
        start_date="2021-01-01T00:00",
        end_date="2021-01-31T23:59",
    )

    df.to_csv(output_dir / "ree_demand_2021_01.csv", index=False)

    print(df.head())
    print(f"Rows downloaded: {len(df)}")