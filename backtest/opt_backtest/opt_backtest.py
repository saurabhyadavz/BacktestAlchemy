import numpy as np
import pandas as pd
import os
from backtest.degen_logger.degen_logger import DegenLogger
from backtest.strategy.strategy import Strategy
from backtest.data.data import DataProducer
from backtest.utils import utils
import pandas_ta as ta
from datetime import datetime, date

from backtest.utils.utils import save_tradebook


class OptBacktest:
    def __init__(self, strategy: Strategy, data: DataProducer):
        self.strategy = strategy
        self.data = data
        self.backtest_logger = DegenLogger(os.path.join(os.getcwd(), self.strategy.strat_name, "backtest.log"))

    def backtest_simple_straddle(self):
        df = self.data.resampled_historical_df
        df["day"] = df["date"].dt.date
        unable_to_trade_days = 0
        tradebook = {'datetime': [], 'price': [], 'side': [], "is_intraday": self.strategy.is_intraday,
                     "unable_to_trade_days": 0}
        date_points = pd.Series(index=df['date'].dt.date.unique(), dtype=object)
        day_groups = df.groupby("day")
        for day, day_df in day_groups:
            entry_close_price = day_df.loc[day_df["date"].dt.time == self.strategy.start_time, "open"].iloc[0]
            atm = int(round(entry_close_price, -2))
            closest_expiry = self.data.get_closet_expiry(self.strategy.instrument, day, week_number=0)
            expiry_comp = self.data.get_expiry_comp_from_date(self.strategy.instrument, closest_expiry)
            option_symbols = []
            atm_ce_symbol = f"{self.data.instrument}{expiry_comp}{atm}CE"
            atm_pe_symbol = f"{self.data.instrument}{expiry_comp}{atm}PE"
            option_symbols.append(atm_ce_symbol)
            option_symbols.append(atm_pe_symbol)

            is_symbol_missing, day_df = utils.get_merged_opt_symbol_df(day_df, option_symbols, day,
                                                                       self.strategy.timeframe)
            if is_symbol_missing:
                unable_to_trade_days += 1
                continue

            ce_stoploss_price, pe_stoploss_price = None, None
            ce_short_price, pe_short_price = None, None
            is_ce_leg_open, is_pe_leg_open = False, False
            is_position = False

            for idx, row in day_df.iterrows():
                curr_time = row["date"].time()

                if curr_time < self.strategy.start_time:
                    continue

                if curr_time >= self.strategy.start_time and not is_position:
                    curr_ce_price = row[f"{atm_ce_symbol}_open"]
                    curr_pe_price = row[f"{atm_pe_symbol}_open"]
                    ce_short_price = curr_ce_price
                    pe_short_price = curr_pe_price
                    tradebook["datetime"].append(day_df.loc[idx, "date"])
                    tradebook["price"].append(ce_short_price)
                    tradebook["side"].append(-1)
                    tradebook["datetime"].append(day_df.loc[idx, "date"])
                    tradebook["price"].append(pe_short_price)
                    tradebook["side"].append(-1)
                    self.backtest_logger.logger.info(
                        f"{row['date']} Shorted {atm_ce_symbol}@{curr_ce_price} & {atm_pe_symbol}@{curr_pe_price}")
                    is_ce_leg_open, is_pe_leg_open = True, True
                    is_position = True
                    ce_stoploss_price = (1 + self.strategy.stoploss_pct) * ce_short_price
                    pe_stoploss_price = (1 + self.strategy.stoploss_pct) * pe_short_price

                if curr_time >= self.strategy.end_time:
                    curr_ce_price = row[f"{atm_ce_symbol}_open"]
                    curr_pe_price = row[f"{atm_pe_symbol}_open"]
                    if is_ce_leg_open:
                        self.backtest_logger.logger.info(
                            f"{row['date']} Day End: Closing {atm_ce_symbol}@{curr_ce_price}")
                        tradebook["datetime"].append(day_df.loc[idx, "date"])
                        tradebook["price"].append(curr_ce_price)
                        tradebook["side"].append(1)
                    if is_pe_leg_open:
                        self.backtest_logger.logger.info(
                            f"{row['date']} Day End: Closing {atm_pe_symbol}@{curr_pe_price}")
                        tradebook["datetime"].append(day_df.loc[idx, "date"])
                        tradebook["price"].append(curr_pe_price)
                        tradebook["side"].append(1)
                    break

                if row[f"{atm_ce_symbol}_high"] >= ce_stoploss_price and is_ce_leg_open:
                    self.backtest_logger.logger.info(
                        f"{row['date']} CE SL HIT: Closing {atm_ce_symbol}@{ce_stoploss_price}")
                    tradebook["datetime"].append(day_df.loc[idx, "date"])
                    tradebook["price"].append(ce_stoploss_price)
                    tradebook["side"].append(1)
                    is_ce_leg_open = False
                    if is_pe_leg_open and self.strategy.move_sl_to_cost:
                        self.backtest_logger.logger.info(
                            f"{row['date']} And Moving {atm_pe_symbol} to cost")
                        pe_stoploss_price = pe_short_price

                if row[f"{atm_pe_symbol}_high"] >= pe_stoploss_price and is_pe_leg_open:
                    tradebook["datetime"].append(day_df.loc[idx, "date"])
                    tradebook["price"].append(pe_stoploss_price)
                    tradebook["side"].append(1)
                    self.backtest_logger.logger.info(
                        f"{row['date']} PE SL HIT: Closing {atm_pe_symbol}@{pe_stoploss_price}")
                    is_pe_leg_open = False
                    if is_ce_leg_open and self.strategy.move_sl_to_cost:
                        ce_stoploss_price = ce_short_price
                        self.backtest_logger.logger.info(
                            f"{row['date']} And Moving {atm_ce_symbol} to cost")

        tradebook["unable_to_trade_days"] = unable_to_trade_days
        save_tradebook(tradebook, self.strategy.strat_name)

    def backtest_positional_iron_condor(self):
        df = self.data.resampled_historical_df
        df["day"] = df["date"].dt.date
        df["Points"] = 0
        track_pnl = pd.Series(np.nan, index=df['date'].dt.date.unique())
        day_groups = df.groupby("day")
        is_position = False
        iron_condor_dict = {}
        expiry_comp = None
        re_execute = False
        re_execute_count = 0
        is_condition_satisfied = False
        exit_date = None
        runtime_pnl = 0
        trade_taken_count = 0
        for day, day_df in day_groups:
            curr_week_expiry_dt = self.data.get_closet_expiry(self.strategy.instrument, day, week_number=0)
            next_week_expiry_dt = self.data.get_closet_expiry(self.strategy.instrument, day, week_number=1)
            entry_n_days_before_expiry = utils.get_n_days_before_date(self.strategy.trading_days_before_expiry,
                                                                      self.data.trading_days, curr_week_expiry_dt)
            exit_n_days_before_expiry = utils.get_n_days_before_date(self.strategy.trading_days_before_expiry,
                                                                     self.data.trading_days, next_week_expiry_dt)

            # Checks if  current day is the entry day, and we don't have any positions
            if entry_n_days_before_expiry == day and not is_position:
                self.backtest_logger.logger.info(f"{day} conditions satisfied for Entry")
                re_execute_count = 0
                exit_date = exit_n_days_before_expiry
                expiry_comp = self.data.get_expiry_comp_from_date(self.strategy.instrument, next_week_expiry_dt)
                self.backtest_logger.logger.info(f"{day}: {expiry_comp}")
                is_condition_satisfied = True

            if not is_condition_satisfied and not is_position:
                continue

            if is_position:
                self.backtest_logger.logger.info(f"{day}: Getting OPT Symbols ")
                is_symbol_missing, day_df = utils.get_merged_opt_symbol_df(day_df, iron_condor_dict["trading_symbols"],
                                                                           day,
                                                                           self.strategy.timeframe)
                if is_symbol_missing:
                    self.backtest_logger.logger.info(f"{day}: Symbol is missing for day: Trade Life Destroyed")
                    is_position = False
                    is_condition_satisfied = False
                    continue

            day_df = day_df.reset_index(drop=True)
            for idx, row in day_df.iterrows():
                curr_time = day_df.loc[idx, "date"].time()
                curr_atm_strike = int(round(day_df.loc[idx, "close"], -2))

                # If re-execute limit is breached, and we don't have any positions and no trade was taken till now
                # then continue
                if re_execute_count == self.strategy.re_execute_count and not is_position and trade_taken_count > 0:
                    continue

                # If it's the strategy start time, and we don't have any positions then take entry
                # or if we want to execute, and we don't have any positions then re-execute
                if (curr_time >= self.strategy.start_time and not is_position) or (re_execute and not is_position):
                    self.backtest_logger.logger.info(f"{day}:  {curr_time} Taking Entry ")
                    iron_condor_dict = utils.generate_iron_condor_strikes_and_symbols(
                        self.strategy.instrument, curr_atm_strike,
                        self.strategy.how_far_otm_short_point,
                        self.strategy.how_far_otm_hedge_point, expiry_comp
                    )
                    self.backtest_logger.logger.info(f"{day}: Current Strike {curr_atm_strike} ")
                    self.backtest_logger.logger.info(f"{day}:  Entry Condor: {iron_condor_dict} ")
                    is_symbol_missing, day_df = utils.get_merged_opt_symbol_df(day_df,
                                                                               iron_condor_dict["trading_symbols"], day,
                                                                               self.strategy.timeframe)
                    if is_symbol_missing:
                        self.backtest_logger.logger.info(
                            f"{day}:  {curr_time} Symbol is missing: Trade Life Destroyed ")
                        is_position = False
                        re_execute_count = 0
                        re_execute = False
                        is_condition_satisfied = False
                        break
                    ce_short_entry_price = day_df.loc[idx, f"{iron_condor_dict['ce_short_opt_symbol']}_close"]
                    pe_short_entry_price = day_df.loc[idx, f"{iron_condor_dict['pe_short_opt_symbol']}_close"]
                    ce_hedge_entry_price = day_df.loc[idx, f"{iron_condor_dict['ce_hedge_opt_symbol']}_close"]
                    pe_hedge_entry_price = day_df.loc[idx, f"{iron_condor_dict['pe_hedge_opt_symbol']}_close"]
                    total_entry_price = (-1 * ce_short_entry_price + -1 * pe_short_entry_price +
                                         1 * ce_hedge_entry_price + 1 * pe_hedge_entry_price)
                    trade_taken_count += 1
                    runtime_pnl += total_entry_price
                    is_position = True
                    if re_execute:
                        re_execute = False
                        re_execute_count += 1

                # If we have position then check if any of strike becomes atm then exit
                if is_position:
                    if (iron_condor_dict["ce_short_strike"] <= curr_atm_strike
                            or iron_condor_dict["pe_short_strike"] >= curr_atm_strike or
                            iron_condor_dict["ce_hedge_strike"] <= curr_atm_strike or
                            iron_condor_dict["pe_hedge_strike"] >= curr_atm_strike or
                            (day == exit_date and curr_time >= self.strategy.end_time)):
                        ce_short_exit_price = day_df.loc[idx, f"{iron_condor_dict['ce_short_opt_symbol']}_close"]
                        pe_short_exit_price = day_df.loc[idx, f"{iron_condor_dict['pe_short_opt_symbol']}_close"]
                        ce_hedge_exit_price = day_df.loc[idx, f"{iron_condor_dict['ce_hedge_opt_symbol']}_close"]
                        pe_hedge_exit_price = day_df.loc[idx, f"{iron_condor_dict['pe_hedge_opt_symbol']}_close"]
                        total_exit_price = (1 * ce_short_exit_price + 1 * pe_short_exit_price +
                                            -1 * ce_hedge_exit_price + -1 * pe_hedge_exit_price)
                        runtime_pnl += total_exit_price
                        is_position = False
                        self.backtest_logger.logger.info(f"{day}:  {curr_time} Taking Exit")
                        # Re-Execute condition
                        if self.strategy.re_execute_count and self.strategy.re_execute_count != re_execute_count:
                            re_execute = True
                        # If its exit day and current time is more than end time then no need of re-execute
                        if day == exit_date and curr_time >= self.strategy.end_time:
                            re_execute = False

                # Update trade life pnl if re-execute count equal to required re-execute count, and we don't have any
                # positions or day is exit day, and we don't have any positions
                if ((
                        re_execute_count == self.strategy.re_execute_count and day <= exit_date and not is_position and trade_taken_count > 0)
                        or (day == exit_date and not is_position and curr_time >= self.strategy.end_time)):
                    self.backtest_logger.logger.info(f"{day}:  {curr_time} Trade Life End")
                    track_pnl.loc[day] = -1 * runtime_pnl
                    runtime_pnl = 0
                    if day == exit_date:
                        self.backtest_logger.logger.info(f"{day}:  {curr_time} Initiating Trade Life")
                        exit_date = exit_n_days_before_expiry
                        trade_taken_count = 0
                        re_execute_count = 0
                        re_execute = False
                        expiry_comp = self.data.get_expiry_comp_from_date(self.strategy.instrument, next_week_expiry_dt)

        track_pnl.dropna(inplace=True)
        return track_pnl

    def backtest_combined_premium_vwap(self):
        df = self.data.resampled_historical_df
        df["day"] = df["date"].dt.date
        tradebook = {'datetime': [], 'price': [], 'side': [], 'traded_quantity': [], "is_intraday": self.strategy.is_intraday,
                     "unable_to_trade_days": 0}
        day_groups = df.groupby("day")
        unable_to_trade_days = 0

        for day, day_df in day_groups:
            # Getting expiry
            curr_week_expiry_dt = self.data.get_closet_expiry(self.strategy.instrument, day,
                                                              week_number=self.strategy.expiry_week)
            expiry_comp = self.data.get_expiry_comp_from_date(self.strategy.instrument, curr_week_expiry_dt)
            entry_close_price = day_df.loc[day_df["date"].dt.time == self.strategy.start_time, "close"].iloc[0]
            entry_atm = int(round(entry_close_price, -2))
            self.backtest_logger.logger.info(f"Start: {day}, atm: {entry_atm}")
            opt_symbols = utils.generate_opt_symbols_from_strike(self.strategy.instrument, entry_atm,
                                                                 expiry_comp, "ATM", "OTM1", "OTM2", "OTM3")
            self.backtest_logger.logger.info(f"symbols: {opt_symbols}")
            is_symbol_missing, day_df = utils.get_combined_premium_df_from_trading_symbols(day_df, opt_symbols,
                                                                                           day,
                                                                                           self.strategy.opt_timeframe)

            if is_symbol_missing:
                with open(os.path.join(os.getcwd(), self.strategy.strat_name, "missing_symbols.txt"), "w") as f:
                    f.write(f"{day}\n")
                unable_to_trade_days += 1
                continue
            day_df.set_index("date", inplace=True)
            # Calculate vwap on combined premium
            day_df["vwap"] = ta.vwap(high=day_df["combined_premium_high"],
                                     low=day_df["combined_premium_low"],
                                     close=day_df["combined_premium_close"],
                                     volume=day_df["combined_premium_volume"])
            day_df["vwap_buffer"] = day_df["vwap"] * (1 + self.strategy.buffer)
            day_df = day_df.reset_index()
            is_position = False
            running_pnl_points = 0
            running_pnl_rupees = 0
            total_trades = 0
            quantity = self.strategy.position_size
            is_half_squared_off = False
            for idx, row in day_df.iterrows():
                curr_time = day_df.loc[idx, "date"].time()
                if curr_time >= self.strategy.end_time:
                    if is_position:
                        self.backtest_logger.logger.info(
                            f"{curr_time} Exit: End of trade px@{day_df.loc[idx, 'combined_premium_close']} qty:{quantity}"
                        )
                        tradebook["datetime"].append(day_df.loc[idx, "date"])
                        tradebook["price"].append(day_df.loc[idx, "combined_premium_close"])
                        tradebook["side"].append(1)
                        tradebook["traded_quantity"].append(quantity)
                        total_trades += 1
                    break
                if curr_time >= self.strategy.start_time and not is_position:
                    if day_df.loc[idx, "combined_premium_close"] < day_df.loc[idx, "vwap_buffer"]:
                        curr_vwap = day_df.loc[idx, 'vwap_buffer']
                        short_price = day_df.loc[idx, "combined_premium_close"]
                        tradebook["datetime"].append(day_df.loc[idx, "date"])
                        tradebook["price"].append(short_price)
                        tradebook["side"].append(-1)
                        tradebook["traded_quantity"].append(quantity)
                        running_pnl_points += short_price * -1
                        running_pnl_rupees += short_price * -1 * quantity
                        is_position = True
                        self.backtest_logger.logger.info(
                            f"{curr_time} Entry :  px@{short_price}, vwap:{curr_vwap} qty:{quantity}"
                        )

                if is_position:
                    if day_df.loc[idx, "combined_premium_close"] > day_df.loc[idx, "vwap_buffer"]:
                        curr_vwap = day_df.loc[idx, 'vwap_buffer']
                        buy_price = day_df.loc[idx, "combined_premium_close"]
                        tradebook["datetime"].append(day_df.loc[idx, "date"])
                        tradebook["price"].append(buy_price)
                        tradebook["side"].append(1)
                        tradebook["traded_quantity"].append(quantity)
                        running_pnl_points += buy_price * 1
                        running_pnl_rupees += buy_price * 1 * quantity
                        self.backtest_logger.logger.info(f"Current PNL {-1 * running_pnl_rupees}")
                        is_position = False
                        self.backtest_logger.logger.info(
                            f"{curr_time} Exit : px@{buy_price}, vwap: {curr_vwap}, qty: {quantity}")
                        quantity = self.strategy.position_size

                    if is_position:
                        # Check MTM in points stoploss
                        curr_close_price = day_df.loc[idx, "combined_premium_close"]
                        curr_pnl_points = (curr_close_price * 1 + running_pnl_points) * -1
                        if self.strategy.stoploss_mtm_points:
                            if curr_pnl_points <= self.strategy.stoploss_mtm_points:
                                tradebook["datetime"].append(day_df.loc[idx, "date"])
                                tradebook["price"].append(curr_close_price)
                                tradebook["side"].append(1)
                                tradebook["traded_quantity"].append(quantity)
                                self.backtest_logger.logger.info(
                                    f"{curr_time} MTM POINT SL : px@{curr_close_price}, qty: {quantity}")
                                break
                        # Check MTM in rupees stoploss
                        if self.strategy.stoploss_mtm_rupees:
                            curr_pnl_rupees = (curr_close_price * 1 * quantity + running_pnl_rupees) * -1
                            if curr_pnl_rupees <= self.strategy.stoploss_mtm_rupees:
                                self.backtest_logger.logger.info(
                                    f"{curr_time}: MTM RUPEES SL HIT: px@{curr_close_price}, qty: {quantity}:"
                                    f" C.PNL: {curr_pnl_rupees}")
                                tradebook["datetime"].append(day_df.loc[idx, "date"])
                                tradebook["price"].append(curr_close_price)
                                tradebook["side"].append(1)
                                tradebook["traded_quantity"].append(quantity)
                                break
                        # Check avg MTM reached
                        if self.strategy.close_half_on_mtm_rupees and not is_half_squared_off:
                            curr_pnl_rupees = (curr_close_price * 1 * quantity + running_pnl_rupees) * -1
                            if curr_pnl_rupees >= self.strategy.close_half_on_mtm_rupees:
                                self.backtest_logger.logger.info(
                                    f"{curr_time} Square off half Qty: px@{curr_close_price} qty: {quantity/2}:"
                                    f" C.PNL: {curr_pnl_rupees}"
                                )
                                tradebook["datetime"].append(day_df.loc[idx, "date"])
                                tradebook["price"].append(curr_close_price)
                                tradebook["side"].append(1)
                                tradebook["traded_quantity"].append(quantity/2)
                                is_half_squared_off = True
                                quantity = quantity / 2
                                running_pnl_points += curr_close_price * 1
                                running_pnl_rupees += curr_close_price * 1 * quantity

        tradebook["unable_to_trade_days"] = unable_to_trade_days
        save_tradebook(tradebook, self.strategy.strat_name)
