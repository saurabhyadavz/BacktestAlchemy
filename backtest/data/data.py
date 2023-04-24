from datetime import datetime
from typing import Union
import pandas as pd
from backtest.utils import utils
import requests


class DataProducer:
    def __init__(self, instrument: str, from_date: Union[str, datetime.date],
                 to_date: Union[str, datetime.date], timeframe: str):
        self.instrument = instrument
        self.from_date = from_date
        self.to_date = to_date
        self.timeframe = timeframe
        self.historical_df = self._fetch_spot_historical_data()
        self.resampled_historical_df = self._resample_ohlc_df()

    def _fetch_spot_historical_data(self) -> pd.DataFrame:
        """
        Fetches Spot data of given instrument(NIFTY/BANKNIFTY)
        Args:
            None

        Returns:
            pd.Dataframe: returns historical data dataframe
        """
        if type(self.from_date) == datetime.date:
            self.from_date = utils.datetime_to_str(self.from_date)
        if type(self.to_date) == datetime.date:
            self.to_date = utils.datetime_to_str(self.to_date)
        request_url = f"{utils.BACKEND_URL}/instruments/historical/{self.instrument}/{self.from_date}/{self.to_date}"
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
                print(f"Error fetching data: {response['message']}")
                return pd.DataFrame()
        response = response.json()
        data = response["data"]
        if response["message"] == "success":
            df = pd.DataFrame(data)
            df["date"] = pd.to_datetime(df["date"])
            df.fillna(method='ffill', inplace=True)
            return df
        else:
            print(f"Error fetching data: {response['message']}")
            return pd.DataFrame()

    def _resample_ohlc_df(self) -> pd.DataFrame:
        """
        Resamples OHLC Dataframe of 1 minute to given timeframe
        Args:
            None
        Returns:
            pd.DataFrame: returns resampled dataframe
        """
        try:
            df_copy = self.historical_df.copy(deep=True)
            df = df_copy
            df["day"] = df.apply(lambda x: x.date.date(), axis=1)
            df.set_index("date", inplace=True)
            day_groups = df.groupby("day")
            resampled_dfs = []
            for day, day_df in day_groups:
                resampled_df = day_df.resample(self.timeframe, origin="start").agg({
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
            print(f"Error occurred while resampling OHLC df [{self.historical_df}: {str(e)}]")
            return pd.DataFrame()
