import typing
from datetime import datetime
from typing import Union
import pandas as pd
from backtest.utils import utils
import requests

from backtest.utils.utils import BANKNIFTY_SYMBOL, NIFTY_SYMBOL


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
            start_time = f'{day} 09:15:00'
            end_time = f'{day} 15:29:00'
            dt_index = pd.date_range(start=start_time, end=end_time, freq='1T')
            df_range = pd.DataFrame(index=dt_index)
            df_range = df_range.rename_axis("date")
            df_range = df_range.merge(day_df, how='left', left_index=True, right_index=True)
            df_range.fillna(method='ffill', inplace=True)
            resample_cols = {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last"
            }
            if "volume" in df_range.columns:
                resample_cols["volume"] = "sum"
            resampled_df = df_range.resample(timeframe, origin="start").agg(resample_cols)
            resampled_df.reset_index(inplace=True)
            resampled_df.fillna(method='ffill', inplace=True)
            resampled_dfs.append(resampled_df)
        combined_df = pd.concat(resampled_dfs, ignore_index=True)
        return combined_df
    except Exception as e:
        print(f"Error occurred while resampling OHLC df [{df}: {str(e)}]")
        return pd.DataFrame()


def get_all_expiry_info(index: str) -> list[dict[typing.Any]]:
    """
    Functions fetches all the expiry dates
        index(str): instrument name(BANKNIFTY/NIFTY)
    Returns:
        list[dict[typing.Any]]: returns a list of dictionary containing(index, expiry_comp, expiry_date)
    """
    url = f"{utils.BACKEND_URL}/nse/get_expiry/{index}"
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


def get_expiry_dates(list_of_expiry_info_dict: list[dict[typing.Any, typing.Any]]):
    expiry_dates_list = [datetime.strptime(d['expiry_date'], '%Y-%m-%d').date()
                         for d in list_of_expiry_info_dict]
    expiry_dates_list = sorted(expiry_dates_list)
    return expiry_dates_list


def get_expiry_comp_dict(list_of_expiry_info_dict: list[dict[typing.Any, typing.Any]]):
    expiry_comp_dict = {
        datetime.strptime(d['expiry_date'], '%Y-%m-%d').date(): d['expiry_comp']
        for d in list_of_expiry_info_dict
    }
    return expiry_comp_dict


def fetch_options_data_and_resample(opt_symbol, start_date, end_date, timeframe):
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
            resampled_df = resampled_df.rename(columns={"close": f"{opt_symbol}_close",
                                                        "open": f"{opt_symbol}_open",
                                                        "high": f"{opt_symbol}_high",
                                                        "low": f"{opt_symbol}_low",
                                                        "volume": f"{opt_symbol}_volume"})
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
        resampled_df = resampled_df.rename(columns={"close": f"{opt_symbol}_close",
                                                    "open": f"{opt_symbol}_open",
                                                    "high": f"{opt_symbol}_high",
                                                    "low": f"{opt_symbol}_low",
                                                    "volume": f"{opt_symbol}_volume"})
        return resampled_df
    else:
        print(f"Error occurred while getting option data for {opt_symbol} for {start_date}")
        return pd.DataFrame()


def get_trading_days():
    url = f"{utils.BACKEND_URL}/nse/get_trading_days/"
    response = requests.get(url)
    response = response.json()
    if response["message"] == "success":
        data = response["data"]
        trading_days = [datetime.strptime(x["Date"], "%Y-%m-%d").date() for x in data]
        return trading_days
    else:
        print(f"Error occurred in func(get_trading_days)")
        return []


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
        self.bnf_list_of_expiry_info_dict = get_all_expiry_info(BANKNIFTY_SYMBOL)
        # self.nf_list_of_expiry_info_dict = get_all_expiry_info(NIFTY_SYMBOL)
        self.bnf_expiry_dates_list = get_expiry_dates(self.bnf_list_of_expiry_info_dict)
        # self.nf_expiry_dates_list = get_expiry_dates(self.nf_list_of_expiry_info_dict)
        self.bnf_expiry_comp_dict = get_expiry_comp_dict(self.bnf_list_of_expiry_info_dict)
        # self.nf_expiry_comp_dict = get_expiry_comp_dict(self.nf_list_of_expiry_info_dict)
        self.trading_days = get_trading_days()

    def get_closet_expiry(self, index: str, dt: datetime.date, week_number: int = 0) -> datetime.date:
        """
        Returns the closest expiry given expiry(current week, next week and so on...)
        Args:
            index(str): instrument(BANKNIFTY/NIFTY)
            dt(datetime.date): given date
            week_number(int): given week number to get the expiry date

        Returns:
            datetime.date: returns the closet expiry date based on the given week number(current week(0), next week(0) and so on...)
        """
        count = 0
        expiry_dates_list = self.bnf_expiry_dates_list
        if index == "NIFTY":
            expiry_dates_list = self.nf_expiry_dates_list
        for expiry_dt in expiry_dates_list:
            if expiry_dt >= dt:
                if week_number == count:
                    return expiry_dt
                count += 1
        print("Error: Couldn't find closest expiry date")
        return None

    def get_expiry_comp_from_date(self, index: str, expiry_date: datetime.date) -> str:
        """
        Returns expiry component for given expiry date
        Args:
            index(str): instrument (ex:BANKNIFTY/NIFTY)
            expiry_date(datetime.date): given expiry date

        Returns:
            str: returns a string expiry component
        """
        try:
            expiry_comp_dict = self.bnf_expiry_comp_dict
            if index == "NIFTY":
                expiry_comp_dict = self.nf_expiry_comp_dict
            return expiry_comp_dict[expiry_date]
        except TypeError as e:
            print(f"Error in func(get_expiry_comp_from_date): {e}")

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
