import requests
import pandas as pd
from pathlib import Path


def download_open_meteo_weather(
    latitude: float,
    longitude: float,
    start_date: str,
    end_date: str,
    location_name: str,
) -> pd.DataFrame:
    """
    Download historical hourly weather data from Open-Meteo.
    """

    url = "https://archive-api.open-meteo.com/v1/archive"

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": [
            "temperature_2m",
            "relative_humidity_2m",
            "precipitation",
            "wind_speed_10m",
            "shortwave_radiation",
        ],
        "timezone": "Europe/Madrid",
    }

    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()

    data = response.json()

    df = pd.DataFrame(data["hourly"])
    df = df.rename(columns={"time": "timestamp"})
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    rename_map = {
        col: f"{col}_{location_name}"
        for col in df.columns
        if col != "timestamp"
    }

    df = df.rename(columns=rename_map)

    return df


if __name__ == "__main__":
    output_dir = Path("data/raw")
    output_dir.mkdir(parents=True, exist_ok=True)

    madrid = download_open_meteo_weather(
        latitude=40.4168,
        longitude=-3.7038,
        start_date="2021-01-01",
        end_date="2021-01-31",
        location_name="madrid",
    )

    madrid.to_csv(output_dir / "weather_madrid_2021_01.csv", index=False)

    print(madrid.head())
    print(f"Rows downloaded: {len(madrid)}")