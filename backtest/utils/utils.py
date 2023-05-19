import typing
from datetime import datetime, time, date
from typing import Union

import numpy as np

from backtest.data import data
import os
import pandas as pd

BACKEND_URL = "http://127.0.0.1:8000"
YY_MM_DD_FORMAT = "%Y-%m-%d"
BNF_WEEKLY_EXP_START_DATE = date(2016, 6, 27)
NF_WEEKLY_EXP_START_DATE = date(2019, 2, 11)
BANKNIFTY_SYMBOL = "BANKNIFTY"
NIFTY_SYMBOL = "NIFTY"
BANKNIFTY_LOT_SIZE = 25
NIFTY_LOT_SIZE = 50
BANKNIFTY_MARGIN_REQUIRED = 180000
NIFTY_MARGIN_REQUIRED = 130000
NOTIONAL_VALUE_ASSUMED = 1000000
BUY = "BUY"
SELL = "SELL"
OPTION_TYPE_CE = "CE"
OPTION_TYPE_PE = "PE"
INT_MAX = 1e18
INT_MIN = -1e18


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


def get_merged_opt_symbol_df(df: pd.DataFrame, trading_symbols: list[str], curr_date: datetime.date,
                             timeframe: str) -> tuple[bool, pd.DataFrame]:
    """
    Merges option symbol close price with given df
    Args:
        df(pd.DataFrame): given dataframe
        trading_symbols(list[str]): iron condor info dictionary
        curr_date(datetime.date): current date
        timeframe(str): timeframe for resampling options df

    Returns:
        tuple[bool, pd.DataFrame]: returns df merged with option symbol close price and option symbol
    """
    is_symbol_missing = False
    for opt_symbol in trading_symbols:
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


def generate_opt_symbols_from_strike(instrument: str, atm_strike: int, expiry_comp: str, *args) -> list[str]:
    """
    Generates strikes based on instrument, atm strike, expiry component and all given strikes in args
    Args:
        instrument(str): instrument name(ex: NIFTY/BANKNIFTY)
        atm_strike(int): at the money strike
        expiry_comp(str): five letter expiry keyword(ex: 16JAN, 16609, 17N23)
        *args(typing.Tuple[typing.Any, ...]): strikes argument like (OTM1, OTM2, ITM1, ITM2, etc)

    Returns:
        list[str]: returns list of option symbols
    """
    opt_symbols = []
    strike_interval = 100
    if instrument == "NIFTY":
        strike_interval = 50
    for arg in args:
        if arg.startswith("ATM"):
            opt_symbols.append(get_opt_symbol(instrument, expiry_comp, atm_strike, "CE"))
            opt_symbols.append(get_opt_symbol(instrument, expiry_comp, atm_strike, "PE"))
        elif arg.startswith("OTM"):
            otm_ce_strike = atm_strike + int(arg[3:]) * strike_interval
            otm_pe_strike = atm_strike - int(arg[3:]) * strike_interval
            opt_symbols.append(get_opt_symbol(instrument, expiry_comp, otm_ce_strike, "CE"))
            opt_symbols.append(get_opt_symbol(instrument, expiry_comp, otm_pe_strike, "PE"))
        elif arg.startswith("ITM"):
            itm_ce_strike = atm_strike - int(arg[3:]) * strike_interval
            itm_pe_strike = atm_strike + int(arg[3:]) * strike_interval
            opt_symbols.append(get_opt_symbol(instrument, expiry_comp, itm_ce_strike, "CE"))
            opt_symbols.append(get_opt_symbol(instrument, expiry_comp, itm_pe_strike, "PE"))
        else:
            raise ValueError(f'Invalid argument: {arg}')
    return opt_symbols


def get_combined_premium_df_from_trading_symbols(df: pd.DataFrame, trading_symbols: list[str], curr_date: datetime.date,
                                                 timeframe: str, is_outerjoin: bool = False) -> tuple[
    bool, pd.DataFrame]:
    """
    Calculates combined premium of given strikes and returns it
    Args:
        df(pd.DataFrame): given dataframe
        trading_symbols(list[str]): iron condor info dictionary
        curr_date(datetime.date): current date
        timeframe(str): timeframe for resampling options df
        is_outerjoin(bool): outer join or not

    Returns:
        tuple[bool, pd.DataFrame]: returns df merged with option symbol close price and option symbol
    """
    close_columns = []
    open_columns = []
    high_columns = []
    low_columns = []
    volume_columns = []
    is_symbol_missing = False
    for opt_symbol in trading_symbols:
        opt_symbol_col = f"{opt_symbol}_close"
        if opt_symbol_col in df.columns:
            continue
        option_df = data.fetch_options_data_and_resample(opt_symbol, curr_date, curr_date, timeframe)
        if option_df.empty:
            is_symbol_missing = True
            with open(os.path.join(os.getcwd(), "missing_symbols.txt"), "a") as f:
                f.write(f"{curr_date} {opt_symbol}\n")
        else:
            close_columns.append(f"{opt_symbol}_close")
            open_columns.append(f"{opt_symbol}_open")
            high_columns.append(f"{opt_symbol}_high")
            low_columns.append(f"{opt_symbol}_low")
            volume_columns.append(f"{opt_symbol}_volume")
            if is_outerjoin:
                df = pd.merge(option_df, df, on="date", how="outer")
            else:
                df = pd.merge(df, option_df, on="date")
    if is_symbol_missing:
        return is_symbol_missing, df
    else:
        df["combined_premium_close"] = df[close_columns].sum(axis=1)
        df["combined_premium_open"] = df[open_columns].sum(axis=1)
        df["combined_premium_high"] = df[high_columns].sum(axis=1)
        df["combined_premium_low"] = df[low_columns].sum(axis=1)
        df["combined_premium_volume"] = df[volume_columns].sum(axis=1)
        return is_symbol_missing, df


def get_instrument_lot_size(instrument: str) -> int:
    """
    Returns instrument lot size
    Args:
        instrument(str): instrument name(BANKNIFTY/NIFTY)

    Returns:
        int: returns instrument lot size
    """
    if instrument == "BANKNIFTY":
        return BANKNIFTY_LOT_SIZE
    return NIFTY_LOT_SIZE


def calculate_leverage(capital: int, lots: int) -> int:
    """
    Calculates leverage for given capital and lots
    Args:
        capital(int): trading capital
        lots(int): trading lots

    Returns:
        int: returns leverage
    """
    lakhs_per_lot = capital / lots
    leverage = int(NOTIONAL_VALUE_ASSUMED / lakhs_per_lot)
    return leverage


def save_tradebook(tradebook_dict: dict[str, typing.Any], strat_name: str):
    """
    Saves Tradebook dictionary as a CSV file
    Args:
        tradebook_dict(dict[str, typing.Any]): tradebook dictionary
        strat_name(str): strategy name

    Returns:

    """
    tradebook_df = pd.DataFrame(tradebook_dict)
    curr_dir = os.path.dirname(os.path.dirname(__file__))
    strategy_dir = os.path.join(curr_dir, strat_name)
    if not os.path.exists(strategy_dir):
        os.mkdir(strategy_dir)
    tradebook_file_path = os.path.join(strategy_dir, f"{strat_name}_tradebook.csv")
    tradebook_df.to_csv(tradebook_file_path, index=False)


def get_vwap(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, achored_time=None) -> pd.Series:
    """
    Returns vwap where priority is given to recent prices
    Args:
        high(pd.Series): candle high series
        low(pd.Series): candle low series
        close(pd.Series): candle close series
        volume(pd.Series): candle volume series
        achored_time: anchored time

    Returns:
        pd.Series: returns vwap series
    """
    col = {
        "high": high,
        "low": low,
        "close": close,
        "volume": volume
    }
    df = pd.DataFrame(col)
    df["date"] = df.index
    df.reset_index(drop=True, inplace=True)
    df["date"] = pd.to_datetime(df["date"])
    N = 5
    multiplier = 2 / (N + 1)
    df["typical_price"] = (df["high"] + df["low"] + df["close"]) / 3
    df["cumm_vol_price"] = np.nan
    df["cumm_vol"] = np.nan
    for i in range(len(df)):
        if df.loc[i, "date"].time() == achored_time:
            df.loc[i, "cumm_vol_price"] = df.loc[i, "typical_price"] * df.loc[i, "volume"]
            df.loc[i, "cumm_vol"] = df.loc[i, "volume"]
        elif df.loc[i, "date"].time() > achored_time:
            df.loc[i, "cumm_vol_price"] = (df.loc[i, "typical_price"] * df.loc[i, "volume"] +
                                           df.loc[i - 1, "cumm_vol_price"])
            df.loc[i, "cumm_vol"] = df.loc[i, "volume"] + df.loc[i - 1, "cumm_vol"]

    df.set_index("date", inplace=True)
    vwap = df["cumm_vol_price"] / df["cumm_vol"]
    return vwap


def add_trade(tradebook: dict[str, typing.Any], running_pnl_points: float,
              running_pnl_rupees: float, curr_dt: pd.Timestamp, price: float, side: str,
              quantity: int) -> tuple[dict[str, typing.Any], float, float]:
    """
    Add trade to tradebook
    Args:
        tradebook(dict[str, typing.Any]): tradebook
        running_pnl_points(float): running pnl in points
        running_pnl_rupees(float): running pnl in rupees
        curr_dt(pd.Timestamp): given date
        price(float): price
        side(str): side(BUY/SELL)
        quantity(int): traded quantity

    Returns:
        dict[str, typing.Any]: returns tradebook dictionary
    """
    if side == BUY:
        side = 1
    else:
        side = -1
    tradebook["datetime"].append(curr_dt)
    tradebook["price"].append(price)
    tradebook["side"].append(side)
    tradebook["traded_quantity"].append(quantity)
    running_pnl_points += price * side
    running_pnl_rupees += price * side * quantity
    return tradebook, running_pnl_points, running_pnl_rupees


def remove_trades(tradebook: dict[str, typing.Any], date_to_remove) -> dict[str, typing.Any]:
    """
    Add trade to tradebook
    Args:
        tradebook(dict[str, typing.Any]): tradebook
        dt: remove rows where date is dt

    Returns:
        dict[str, typing.Any]: returns tradebook dictionary
    """
    indices_to_remove = [i for i, dt in enumerate(tradebook['datetime']) if dt.startswith(date_to_remove)]
    for key in tradebook:
        tradebook[key] = [tradebook[key][i] for i in range(len(tradebook[key])) if i not in indices_to_remove]
    return tradebook
