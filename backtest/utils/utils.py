import typing
from datetime import datetime, time
from typing import Union
from backtest.data import data
import os
import pandas as pd

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


def str_to_time_obj(input_str: str) -> Union[datetime.time, None]:
    """
    Convert string of format "hh:mm" to time object
    Args:
        input_str(str): input time string of format "hh:mm"

    Returns:
        datetime.time: returns time object
    """
    try:
        hours, minutes = map(int, input_str.split(":"))
        time_obj = time(hour=hours, minute=minutes)
    except ValueError:
        print("Invalid input format. Please enter a string in the format 'hh:mm'")
        return None
    return time_obj


def get_opt_symbol(instrument: str, expiry_comp: str, strike: int, opt_type: str) -> str:
    """
    Returns Option symbol given instrument, strike, expiry component and option type(CE/PE)
    Args:
        instrument(str): instrument(ex: NIFTY/BANKNIFTY)
        expiry_comp(str): five letter expiry keyword(ex: 16JAN, 16609, 17N23)
        strike(int): strike
        opt_type(str): option type(CE/PE)

    Returns:
        str: returns option symbol
    """
    opt_symbol = f"{instrument.upper()}{expiry_comp}{strike}{opt_type}"
    return opt_symbol


def generate_iron_condor_strikes_and_symbols(instrument: str, atm_strike: int, how_far_otm_short_point: int,
                                             how_far_otm_hedge_point: int, expiry_comp: str) -> dict[str, typing.Any]:
    """
    Generates strikes, option symbols for iron condor strategy
    Args:
        instrument(str): instrument name(ex: NIFTY/BANKNIFTY)
        atm_strike(int): at the money strike
        how_far_otm_short_point(int): how far from at the money to short
        how_far_otm_hedge_point(int): how far from at the money to buy hedge
        expiry_comp(str): five letter expiry keyword(ex: 16JAN, 16609, 17N23)

    Returns:
        dict[str, typing.Any]: returns dict of option symbols, strikes and list of option symbols
    """
    iron_condor_dict = {}
    iron_condor_dict["ce_short_strike"] = atm_strike + how_far_otm_short_point
    iron_condor_dict["pe_short_strike"] = atm_strike - how_far_otm_short_point
    iron_condor_dict["ce_hedge_strike"] = atm_strike + how_far_otm_hedge_point
    iron_condor_dict["pe_hedge_strike"] = atm_strike - how_far_otm_hedge_point
    iron_condor_dict["ce_short_opt_symbol"] = get_opt_symbol(instrument, expiry_comp,
                                                             iron_condor_dict["ce_short_strike"], "CE")
    iron_condor_dict["pe_short_opt_symbol"] = get_opt_symbol(instrument, expiry_comp,
                                                             iron_condor_dict["pe_short_strike"], "PE")
    iron_condor_dict["ce_hedge_opt_symbol"] = get_opt_symbol(instrument, expiry_comp,
                                                             iron_condor_dict["ce_hedge_strike"], "CE")
    iron_condor_dict["pe_hedge_opt_symbol"] = get_opt_symbol(instrument, expiry_comp,
                                                             iron_condor_dict["pe_hedge_strike"], "PE")
    iron_condor_dict["trading_symbols"] = [iron_condor_dict["ce_short_opt_symbol"],
                                           iron_condor_dict["pe_short_opt_symbol"],
                                           iron_condor_dict["ce_hedge_opt_symbol"],
                                           iron_condor_dict["pe_hedge_opt_symbol"]]
    return iron_condor_dict


def get_merged_opt_symbol_df(df: pd.DataFrame, iron_condor_dict: dict[str, typing.Any],
                             curr_date: datetime.date, timeframe: str) -> tuple[bool, pd.DataFrame]:
    """
    Merges option symbol close price with given df
    Args:
        df(pd.DataFrame): given dataframe
        iron_condor_dict(dict[str, typing.Any]): iron condor info dictionary
        curr_date(datetime.date): current date
        timeframe(str): timeframe for resampling options df

    Returns:
        tuple[bool, pd.DataFrame]: returns df merged with option symbol close price and option symbol
    """
    is_symbol_missing = False
    for opt_symbol in iron_condor_dict["trading_symbols"]:
        opt_symbol_col = f"{opt_symbol}_close"
        if opt_symbol_col in df.columns:
            continue
        option_df = data.fetch_options_data_and_resample(opt_symbol, curr_date, curr_date, timeframe)
        if option_df.empty:
            is_symbol_missing = True
            with open(os.path.join(os.getcwd(), "missing_symbols.txt"), "a") as f:
                f.write(f"{curr_date} {opt_symbol}\n")
        else:
            df = pd.merge(df, option_df, on='date')
    return is_symbol_missing, df


def get_n_days_before_date(n: int, trading_dates: list[datetime.date], date_to_find: datetime.date) -> datetime.date:
    """
    Merges option symbol close price with given df
    Args:
        n(int): n days before given date
        trading_dates(list[datetime.date]):  list of trading dates
        date_to_find(datetime.date): date to find

    Returns:
        datetime.date: returns n days before given date
    """
    left, right = 0, len(trading_dates) - 1
    find_index = -1
    while left <= right:
        mid = (left + right) // 2
        if trading_dates[mid] < date_to_find:
            left = mid + 1
        elif trading_dates[mid] > date_to_find:
            right = mid - 1
        else:
            find_index = mid
            break
    if find_index == -1:
        print(f"{date_to_find} doesn't exist in trading dates list")
        return -1
    else:
        if (find_index - n) >= 0:
            return trading_dates[find_index - n]
        else:
            print(f"{n} days before {date_to_find} doesn't exist")
            return -1

