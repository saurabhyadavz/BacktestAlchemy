from typing import Union
import pandas as pd
import requests
from datetime import datetime

BACKEND_URL = "http://127.0.0.1:8000"
YY_MM_DD_FORMAT = "%Y-%m-%d"


def string_to_datetime(date_string: str) -> datetime:
    """
    Converts a string to a datetime object.
    Args:
        date_string (str): A string representing a date in the format 'YYYY-MM-DD'

    Returns:
        datetime: A datetime object representing the output date.
    """
    try:
        return datetime.strptime(date_string, YY_MM_DD_FORMAT)
    except TypeError as e:
        print(f"Error in func(string_to_datetime): {str(e)}")


def datetime_to_str(date_obj: datetime.date) -> str:
    """
    Converts a datetime object to a string.
    Args:
        date_obj (datetime.date): A datetime.date representing a datetime object

    Returns:
        str: A string representing the output date.
    """
    try:
        return date_obj.strftime(YY_MM_DD_FORMAT)
    except TypeError as e:
        print(f"Error in func(datetime_to_str): {str(e)}")

def fetch_spot_historical_data(instrument: str, from_date: Union[str, datetime.date], to_date: Union[str, datetime.date]) -> pd.DataFrame:
    """
    Fetches Spot data of given instrument(NIFTY/BANKNIFTY)
    Args:
        instrument(str): instrument name (ex: NIFTY/BANKNIFTY)
        from_date(Union[str, datetime.date]): from date(of type str(format YYYY-MM-DD) or datetime.date object)
        to_date(Union[str, datetime.date]): to date(of type str(format YYYY-MM-DD) or datetime.date object)

    Returns:
        pd.Dataframe: returns historical data dataframe
    """
    if type(from_date) == datetime.date:
        from_date = datetime_to_str(from_date)
    if type(to_date) == datetime.date:
        to_date = datetime_to_str(to_date)
    request_url = f"{BACKEND_URL}/instruments/historical/{instrument}/{from_date}/{to_date}"
    response = requests.get(request_url, params={"spot": "true"})
    data = None
    if response.status_code == 307:
        redirect_url = response.headers['Location']
        print(f"Redirecting to: {redirect_url}")
        response = requests.get(redirect_url, params={"spot": "true"})
        response = response.json()
        if response["message"] == "success":
            data = response["data"]
            df = pd.DataFrame(data)
            df["date"] = pd.to_datetime(df["date"])
            df.fillna(method='ffill', inplace=True)
            return df
        else:
            print(f"Error fetching data for {response['message']}")
            return pd.DataFrame()
    response = response.json()
    data = response["data"]
    if response["message"] == "success":
        df = pd.DataFrame(data)
        df["date"] = pd.to_datetime(df["date"])
        df.fillna(method='ffill', inplace=True)
        return df
    else:
        print(f"Error fetching data for {response['message']}")
        return pd.DataFrame()


def resample_ohlc_df(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """
    Resamples OHLC Dataframe of 1 minute to given timeframe
    Args:
        df(pd.Dataframe): 1 min OHLC dataframe
        timeframe(str): Timeframe to resample of format(1min, 3min, 5min, 10min, D)

    Returns:
        pd.DataFrame: returns resampled dataframe
    """
    try:
        df_copy = df.copy(deep=True)
        df = df_copy
        df["day"] = df.apply(lambda x: x.date.date(), axis=1)
        df.set_index("date", inplace=True)
        day_groups = df.groupby("day")
        resampled_dfs = []
        for day, day_df in day_groups:
            resampled_df = day_df.resample(timeframe, origin="start").agg({
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last"
            })
            resampled_df.reset_index(inplace=True)
            resampled_df.fillna(method='ffill', inplace=True)
            resampled_dfs.append(resampled_df)
        combined_df = pd.concat(resampled_dfs, ignore_index=True)
        return combined_df
    except Exception as e:
        print(f"Error occurred while resampling OHLC df {timeframe}: {str(e)}")
        return pd.DataFrame()
