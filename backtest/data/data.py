import typing
from datetime import datetime
from typing import Union
import pandas as pd
from backtest.utils import utils
import requests


def resample_ohlc_df(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """
    Resamples OHLC Dataframe of 1 minute to given timeframe
    Args:
        df(pd.Dataframe): input dataframe
        timeframe(str): timeframe to resample data(1min, 5min, 15min, 30min, 60min, D)
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
        print(f"Error occurred while resampling OHLC df [{df}: {str(e)}]")
        return pd.DataFrame()

def get_all_expiry_dates() -> list[dict[typing.Any]]:
    """
    Functions fetches all the expiry dates
    Returns:
        list[dict[typing.Any]]: returns a list of dictionary containing(index, expiry_str, expiry_date)
    """
    url = f"{utils.BACKEND_URL}/nse/get_expiry/"
    response = requests.get(url)
    if response.status_code == 307:
        redirect_url = response.headers['Location']
        print(f"Redirecting to: {redirect_url}")
        response = requests.get(redirect_url)
        response = response.json()
        if response["message"] == "success":
            data = response["data"]
            return data
        else:
            print(f"Error fetching expiry dates")
            return []
    data = response.json()["data"]
    return data


class DataProducer:
    def __init__(self, instrument: str, from_date: Union[str, datetime.date],
                 to_date: Union[str, datetime.date], timeframe: str, expiry_week: int = 0):
        self.instrument = instrument
        self.from_date = from_date
        self.to_date = to_date
        self.timeframe = timeframe
        self.historical_df = self._fetch_spot_historical_data()
        self.resampled_historical_df = resample_ohlc_df(self.historical_df, self.timeframe)
        self.expiry_week = expiry_week
        self.list_of_expiry_dicts = get_all_expiry_dates()
        self.historical_df_with_exp_comp = self.get_ohlc_with_expiry_comp(self.resampled_historical_df)

    def get_greater_closet_expiry(self, dt: datetime.date, expiry_dates: list[datetime.date]) -> datetime.date:
        """
        Returns the closest expiry given expiry(current week, next week and so on...)
        Args:
            dt(datetime.date): given date
            expiry_dates(list[datetime.date]): list of all expiry dates

        Returns:
            datetime.date: returns the closet expiry date based on the given expiry(current week, next week and so on...)
        """
        count = 0
        for expiry_dt in expiry_dates:
            if expiry_dt >= dt:
                if self.expiry_week == count:
                    return expiry_dt
                count += 1
        print("Error: Couldn't find closest expiry date")
        return None

    def get_expiry_comp_from_date(self, dt: datetime.date, expiry_dates: list[datetime.date],
                                  expiry_date_comp_dict: dict[str, datetime.date]) -> str:
        """
        Returns expiry component by finding the closest expiry week given date
        Args:
            dt(datetime.date): given date
            expiry_dates(list[datetime.date]): list of all expiry dates
            expiry_date_comp_dict(dict[str, datetime.date]):

        Returns:
            str: returns a string expiry component
        """
        try:
            closet_expiry_date = self.get_greater_closet_expiry(dt, expiry_dates)
            return expiry_date_comp_dict[closet_expiry_date]
        except TypeError as e:
            print(f"Error in func(get_expiry_comp_from_date): {e}")

    def get_ohlc_with_expiry_comp(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add expiry_comp column in df given expiry week
        Args:
            df(pd.Dataframe): given OHLC dataframe

        Returns:
            pd.DataFrame: returns dataframe with expiry_comp col added in it
        """
        try:
            expiry_dates_list = [datetime.strptime(d['expiry_date'], '%Y-%m-%d').date()
                                 for d in self.list_of_expiry_dicts]
            expiry_date_comp_dict = {
                datetime.strptime(d['expiry_date'], '%Y-%m-%d').date(): d['expiry_str']
                for d in self.list_of_expiry_dicts
            }
            df["expiry_comp"] = df.apply(lambda row: self.get_expiry_comp_from_date(
                row["date"].date(),
                expiry_dates_list,
                expiry_date_comp_dict
            ), axis=1)
            return df
        except Exception as e:
            print(f"An error occurred: {e}")
            return pd.DataFrame()

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

    def fetch_options_data_and_resample(self, opt_symbol, start_date, end_date, timeframe):
        url = f"{utils.BACKEND_URL}/instruments/historical/{opt_symbol}/{start_date}/{end_date}/"
        response = requests.get(url, params={"spot": "false"})
        data = None
        if response.status_code == 307:
            redirect_url = response.headers['Location']
            print(f"Redirecting to: {redirect_url}")
            response = requests.get(redirect_url, params={"spot": "false"})
            response = response.json()
            if response["message"] == "success":
                data = response["data"]
                df = pd.DataFrame(data)
                df["date"] = pd.to_datetime(df["date"])
                resampled_df = resample_ohlc_df(df, timeframe)
                resampled_df.drop(["open", "high", "low"], axis=1, inplace=True)
                resampled_df = resampled_df.rename(columns={"close": f"{opt_symbol}_close"})
                return resampled_df
            else:
                print(f"Error fetching data for: {opt_symbol}")
                return pd.DataFrame()
        response = response.json()
        if response["message"] == "success":
            data = response["data"]
            df = pd.DataFrame(data)
            df["date"] = pd.to_datetime(df["date"])
            resampled_df = resample_ohlc_df(df, timeframe)
            resampled_df.drop(["open", "high", "low"], axis=1, inplace=True)
            resampled_df = resampled_df.rename(columns={"close": f"{opt_symbol}_close"})
            return resampled_df
        else:
            print(f"Error occurred while getting option data for {opt_symbol} for {start_date}")
            return pd.DataFrame()
