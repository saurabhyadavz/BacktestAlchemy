import numpy as np
import pandas as pd
import os
from backtest.degen_logger.degen_logger import DegenLogger
from backtest.strategy.strategy import Strategy
from backtest.data.data import DataProducer, CandleData, resample_ohlc_df
from backtest.utils import utils
import pandas_ta as ta
from datetime import datetime, date, timedelta
from backtest.utils.utils import save_tradebook, BUY, SELL, OPTION_TYPE_CE, OPTION_TYPE_PE


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
        tradebook = {'datetime': [], 'price': [], 'side': [], 'traded_quantity': [],
                     "is_intraday": self.strategy.is_intraday,
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
                                                                 expiry_comp, "OTM1", "OTM2", "OTM3")
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
                        is_position = False
                        self.backtest_logger.logger.info(
                            f"{curr_time} Exit : px@{buy_price}, vwap: {curr_vwap}, qty: {quantity} PNL: {running_pnl_rupees * -1}")
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
                            # Check if MTM SL hit at close
                            curr_pnl_rupees = (curr_close_price * 1 * quantity + running_pnl_rupees) * -1
                            if curr_pnl_rupees <= self.strategy.stoploss_mtm_rupees:
                                self.backtest_logger.logger.info(
                                    f"{curr_time}: MTM RUPEES SL HIT: CLOSEpx@{curr_close_price}, qty: {quantity} C.PNL: {curr_pnl_rupees}")
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
                                    f"{curr_time} Square off half Qty: px@{curr_close_price} qty: {quantity / 2}:"
                                    f" C.PNL: {curr_pnl_rupees}"
                                )
                                tradebook["datetime"].append(day_df.loc[idx, "date"])
                                tradebook["price"].append(curr_close_price)
                                tradebook["side"].append(1)
                                tradebook["traded_quantity"].append(quantity / 2)
                                is_half_squared_off = True
                                quantity = quantity / 2
                                running_pnl_points += curr_close_price * 1
                                running_pnl_rupees += curr_close_price * 1 * quantity

        tradebook["unable_to_trade_days"] = unable_to_trade_days
        save_tradebook(tradebook, self.strategy.strat_name)

    def backtest_itm_vwap(self):
        df = self.data.resampled_historical_df
        df["day"] = df["date"].dt.date
        tradebook = {'datetime': [], 'price': [], 'side': [], 'traded_quantity': [],
                     "is_intraday": self.strategy.is_intraday,
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
            ce_opt_symbol = utils.get_opt_symbol(self.strategy.instrument, expiry_comp, entry_atm - 500, "CE")
            pe_opt_symbol = utils.get_opt_symbol(self.strategy.instrument, expiry_comp, entry_atm + 500, "PE")
            opt_symbols = [ce_opt_symbol, pe_opt_symbol]
            self.backtest_logger.logger.info(f"Start: {day}, atm: {entry_atm}")
            self.backtest_logger.logger.info(f"symbols: {opt_symbols}")
            is_symbol_missing, day_df = utils.get_merged_opt_symbol_df(day_df, opt_symbols, day,
                                                                       self.strategy.opt_timeframe)

            if is_symbol_missing:
                with open(os.path.join(os.getcwd(), self.strategy.strat_name, "missing_symbols.txt"), "w") as f:
                    f.write(f"{day}\n")
                unable_to_trade_days += 1
                continue
            day_df.set_index("date", inplace=True)
            # Calculate vwap on combined premium
            day_df[f"{ce_opt_symbol}_vwap"] = ta.vwap(high=day_df[f"{ce_opt_symbol}_high"],
                                                      low=day_df[f"{ce_opt_symbol}_low"],
                                                      close=day_df[f"{ce_opt_symbol}_close"],
                                                      volume=day_df[f"{ce_opt_symbol}_volume"])
            day_df[f"{pe_opt_symbol}_vwap"] = ta.vwap(high=day_df[f"{pe_opt_symbol}_high"],
                                                      low=day_df[f"{pe_opt_symbol}_low"],
                                                      close=day_df[f"{pe_opt_symbol}_close"],
                                                      volume=day_df[f"{pe_opt_symbol}_volume"])
            day_df[f"{ce_opt_symbol}_vwap"] = day_df[f"{ce_opt_symbol}_vwap"] * (1 + self.strategy.buffer)
            day_df[f"{pe_opt_symbol}_vwap"] = day_df[f"{pe_opt_symbol}_vwap"] * (1 + self.strategy.buffer)
            day_df = day_df.reset_index()
            is_ce_position = False
            running_pnl_points = 0
            running_pnl_rupees = 0
            total_trades = 0
            quantity = self.strategy.position_size
            is_half_squared_off = False
            ce_hard_sl, pe_hard_sl = None, None
            prev_row = pd.Series()
            prev_prev_row = pd.Series()
            is_pe_position = False
            ce_buffer_time = None
            pe_buffer_time = None
            is_ce_first_entry = True
            is_pe_first_entry = True
            is_ce_entry_trigger = False
            is_pe_entry_trigger = False
            for idx, row in day_df.iterrows():
                curr_time = day_df.loc[idx, "date"].time()
                if curr_time >= self.strategy.end_time:
                    if is_ce_position:
                        buy_price = day_df.loc[idx, f"{ce_opt_symbol}_close"]
                        tradebook["datetime"].append(day_df.loc[idx, "date"])
                        tradebook["price"].append(buy_price)
                        tradebook["side"].append(1)
                        tradebook["traded_quantity"].append(quantity)
                        running_pnl_points += buy_price * 1
                        running_pnl_rupees += buy_price * 1 * quantity
                        self.backtest_logger.logger.info(
                            f"{curr_time} Exit: {ce_opt_symbol} End of trade px@{day_df.loc[idx, f'{ce_opt_symbol}_close']} qty:{quantity} PNL: {running_pnl_rupees * -1}"
                        )
                        total_trades += 1

                    if is_pe_position:
                        buy_price = day_df.loc[idx, f"{pe_opt_symbol}_close"]
                        tradebook["datetime"].append(day_df.loc[idx, "date"])
                        tradebook["price"].append(buy_price)
                        tradebook["side"].append(1)
                        tradebook["traded_quantity"].append(quantity)
                        running_pnl_points += buy_price * 1
                        running_pnl_rupees += buy_price * 1 * quantity
                        self.backtest_logger.logger.info(
                            f"{curr_time} Exit: {pe_opt_symbol} End of trade px@{buy_price} qty:{quantity} PNL: {running_pnl_rupees * -1}"
                        )
                        total_trades += 1
                    break
                if curr_time >= self.strategy.start_time and not is_ce_position:

                    if (is_ce_entry_trigger and
                            day_df.loc[idx, f"{ce_opt_symbol}_close"] < day_df.loc[idx, f"{ce_opt_symbol}_vwap"]):
                        curr_vwap = day_df.loc[idx, f"{ce_opt_symbol}_vwap"]
                        short_price = day_df.loc[idx, f"{ce_opt_symbol}_open"]
                        tradebook["datetime"].append(day_df.loc[idx, "date"])
                        tradebook["price"].append(short_price)
                        tradebook["side"].append(-1)
                        tradebook["traded_quantity"].append(quantity)
                        running_pnl_points += short_price * -1
                        running_pnl_rupees += short_price * -1 * quantity
                        is_ce_position = True
                        is_ce_entry_trigger = False
                        ce_buffer_time = curr_time

                        if is_ce_first_entry:
                            delta = timedelta(minutes=15)
                            ce_buffer_time = (datetime.combine(date.today(), curr_time) + delta).time()
                            self.backtest_logger.logger.info(f"{curr_time}: CE_BUFFER: {ce_buffer_time}")
                            if not prev_row.empty and f"{ce_opt_symbol}_high" in prev_row:
                                ce_hard_sl = max(prev_row[f"{ce_opt_symbol}_high"], row[f"{ce_opt_symbol}_high"])
                            else:
                                ce_hard_sl = row[f"{ce_opt_symbol}_high"]
                            is_ce_first_entry = False

                        self.backtest_logger.logger.info(
                            f"{curr_time} Entry : {ce_opt_symbol} px@{short_price}, vwap:{curr_vwap} qty:{quantity}"
                        )
                        self.backtest_logger.logger.info(f"HARD SL CE {ce_hard_sl}")

                    if (not is_ce_position and
                            day_df.loc[idx, f"{ce_opt_symbol}_close"] < day_df.loc[idx, f"{ce_opt_symbol}_vwap"]):
                        is_ce_entry_trigger = True
                        self.backtest_logger.logger.info(f"{curr_time}  Trigger: {ce_opt_symbol}")

                if curr_time >= self.strategy.start_time and not is_pe_position:
                    if (is_pe_entry_trigger and
                            day_df.loc[idx, f"{pe_opt_symbol}_open"] < day_df.loc[idx, f"{pe_opt_symbol}_vwap"]):
                        curr_vwap = day_df.loc[idx, f"{pe_opt_symbol}_vwap"]
                        short_price = day_df.loc[idx, f"{pe_opt_symbol}_open"]
                        tradebook["datetime"].append(day_df.loc[idx, "date"])
                        tradebook["price"].append(short_price)
                        tradebook["side"].append(-1)
                        tradebook["traded_quantity"].append(quantity)
                        running_pnl_points += short_price * -1
                        running_pnl_rupees += short_price * -1 * quantity
                        is_pe_position = True
                        pe_buffer_time = curr_time
                        delta = timedelta(minutes=15)

                        if is_pe_first_entry:
                            pe_buffer_time = (datetime.combine(date.today(), curr_time) + delta).time()
                            self.backtest_logger.logger.info(f"{curr_time}: PE_BUFFER: {pe_buffer_time}")
                            if not prev_prev_row.empty and f"{pe_opt_symbol}_high" in prev_prev_row:
                                pe_hard_sl = max(prev_row[f"{pe_opt_symbol}_high"],
                                                 prev_prev_row[f"{pe_opt_symbol}_high"])
                            else:
                                pe_hard_sl = prev_row[f"{pe_opt_symbol}_high"]
                            is_pe_first_entry = False

                        self.backtest_logger.logger.info(
                            f"{curr_time} Entry : {pe_opt_symbol} px@{short_price}, vwap:{curr_vwap} qty:{quantity}"
                        )
                        self.backtest_logger.logger.info(f"HARD SL PE {pe_hard_sl}")

                    if (not is_pe_position and
                            day_df.loc[idx, f"{pe_opt_symbol}_close"] < day_df.loc[idx, f"{pe_opt_symbol}_vwap"]):
                        is_pe_entry_trigger = True
                        self.backtest_logger.logger.info(f"{curr_time}  Trigger: {pe_opt_symbol}")

                if is_ce_position:
                    # If candle high breaches Hard SL, exit
                    if ce_hard_sl and day_df.loc[idx, f"{ce_opt_symbol}_high"] > ce_hard_sl:
                        tradebook["datetime"].append(day_df.loc[idx, "date"])
                        tradebook["price"].append(ce_hard_sl)
                        tradebook["side"].append(1)
                        tradebook["traded_quantity"].append(quantity)
                        self.backtest_logger.logger.info(
                            f"{curr_time} HARD SL HIT : {ce_opt_symbol} px@{ce_hard_sl}, PNL: {running_pnl_rupees * -1}")
                        running_pnl_points += ce_hard_sl * 1
                        running_pnl_rupees += ce_hard_sl * 1 * quantity
                        is_ce_position = False
                        ce_hard_sl = None
                        quantity = self.strategy.position_size

                    # If close > vwap, exit
                    if (curr_time > ce_buffer_time and
                            day_df.loc[idx, f"{ce_opt_symbol}_close"] > day_df.loc[idx, f"{ce_opt_symbol}_vwap"]):
                        curr_vwap = day_df.loc[idx, f"{ce_opt_symbol}_vwap"]
                        buy_price = day_df.loc[idx, f"{ce_opt_symbol}_close"]
                        tradebook["datetime"].append(day_df.loc[idx, "date"])
                        tradebook["price"].append(buy_price)
                        tradebook["side"].append(1)
                        tradebook["traded_quantity"].append(quantity)
                        running_pnl_points += buy_price * 1
                        running_pnl_rupees += buy_price * 1 * quantity
                        is_ce_position = False
                        self.backtest_logger.logger.info(
                            f"{curr_time} Exit : {ce_opt_symbol} px@{buy_price}, vwap: {curr_vwap}, qty: {quantity} PNL: {running_pnl_rupees * -1}")
                        quantity = self.strategy.position_size

                if is_pe_position:
                    # If candle high breaches Hard SL, exit
                    if pe_hard_sl and day_df.loc[idx, f"{pe_opt_symbol}_high"] > pe_hard_sl:
                        tradebook["datetime"].append(day_df.loc[idx, "date"])
                        tradebook["price"].append(pe_hard_sl)
                        tradebook["side"].append(1)
                        tradebook["traded_quantity"].append(quantity)
                        self.backtest_logger.logger.info(
                            f"{curr_time} HARD SL HIT : {pe_opt_symbol} px@{pe_hard_sl}, PNL: {running_pnl_rupees * -1}")
                        running_pnl_points += pe_hard_sl * 1
                        running_pnl_rupees += pe_hard_sl * 1 * quantity
                        is_pe_position = False
                        pe_hard_sl = None
                        quantity = self.strategy.position_size

                    # If close > vwap, exit
                    if (curr_time > pe_buffer_time and
                            day_df.loc[idx, f"{pe_opt_symbol}_close"] > day_df.loc[idx, f"{pe_opt_symbol}_vwap"]):
                        curr_vwap = day_df.loc[idx, f"{pe_opt_symbol}_vwap"]
                        buy_price = day_df.loc[idx, f"{pe_opt_symbol}_close"]
                        tradebook["datetime"].append(day_df.loc[idx, "date"])
                        tradebook["price"].append(buy_price)
                        tradebook["side"].append(1)
                        tradebook["traded_quantity"].append(quantity)
                        running_pnl_points += buy_price * 1
                        running_pnl_rupees += buy_price * 1 * quantity
                        is_pe_position = False
                        self.backtest_logger.logger.info(
                            f"{curr_time} Exit : {pe_opt_symbol} px@{buy_price}, vwap: {curr_vwap}, qty: {quantity} PNL: {running_pnl_rupees * -1}")
                        quantity = self.strategy.position_size

                # Check MTM SL HIT or NOT
                if (is_ce_position or is_pe_position) and self.strategy.stoploss_mtm_rupees:
                    curr_ce_close_px = day_df.loc[idx, f"{ce_opt_symbol}_close"]
                    curr_pe_close_px = day_df.loc[idx, f"{pe_opt_symbol}_close"]
                    curr_pnl = running_pnl_rupees
                    if is_ce_position:
                        curr_pnl += (curr_ce_close_px * 1 * quantity)
                    if is_pe_position:
                        curr_pnl += (curr_pe_close_px * 1 * quantity)
                    curr_pnl *= -1
                    if curr_pnl <= self.strategy.stoploss_mtm_rupees:
                        if is_ce_position:
                            self.backtest_logger.logger.info(
                                f"{curr_time}: MTM RUPEES SL HIT: {ce_opt_symbol} px@{curr_ce_close_px}, qty: {quantity} C.PNL: {curr_pnl}")
                            tradebook["datetime"].append(day_df.loc[idx, "date"])
                            tradebook["price"].append(curr_ce_close_px)
                            tradebook["side"].append(1)
                            tradebook["traded_quantity"].append(quantity)
                            running_pnl_points += curr_ce_close_px * 1
                            running_pnl_rupees += curr_ce_close_px * 1 * quantity
                        if is_pe_position:
                            tradebook["datetime"].append(day_df.loc[idx, "date"])
                            tradebook["price"].append(curr_pe_close_px)
                            tradebook["side"].append(1)
                            tradebook["traded_quantity"].append(quantity)
                            running_pnl_points += curr_pe_close_px * 1
                            running_pnl_rupees += curr_pe_close_px * 1 * quantity
                            self.backtest_logger.logger.info(
                                f"{curr_time} MTM RUPEES SL HIT: {pe_opt_symbol} px@{curr_pe_close_px} qty:{quantity} PNL: {running_pnl_rupees * -1}"
                            )
                        break
                prev_row = row

        tradebook["unable_to_trade_days"] = unable_to_trade_days
        save_tradebook(tradebook, self.strategy.strat_name)

    def backtest_atm_anchored_vwap(self):
        df = self.data.resampled_historical_df
        df["day"] = df["date"].dt.date
        tradebook = {'datetime': [], 'price': [], 'side': [], 'traded_quantity': [],
                     "is_intraday": self.strategy.is_intraday,
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
            ce_opt_symbol = utils.get_opt_symbol(self.strategy.instrument, expiry_comp, entry_atm, "CE")
            pe_opt_symbol = utils.get_opt_symbol(self.strategy.instrument, expiry_comp, entry_atm, "PE")
            opt_symbols = [ce_opt_symbol, pe_opt_symbol]
            self.backtest_logger.logger.info(f"Start: {day}, atm: {entry_atm}")
            self.backtest_logger.logger.info(f"symbols: {opt_symbols}")
            is_symbol_missing, day_df = utils.get_merged_opt_symbol_df(day_df, opt_symbols, day,
                                                                       self.strategy.opt_timeframe)

            if is_symbol_missing:
                with open(os.path.join(os.getcwd(), self.strategy.strat_name, "missing_symbols.txt"), "w") as f:
                    f.write(f"{day}\n")
                unable_to_trade_days += 1
                continue
            day_df.set_index("date", inplace=True)
            # Calculate vwap on combined premium
            day_df[f"{ce_opt_symbol}_vwap"] = utils.get_vwap(high=day_df[f"{ce_opt_symbol}_high"],
                                                             low=day_df[f"{ce_opt_symbol}_low"],
                                                             close=day_df[f"{ce_opt_symbol}_close"],
                                                             volume=day_df[f"{ce_opt_symbol}_volume"],
                                                             achored_time=self.strategy.start_time)
            day_df[f"{pe_opt_symbol}_vwap"] = utils.get_vwap(high=day_df[f"{pe_opt_symbol}_high"],
                                                             low=day_df[f"{pe_opt_symbol}_low"],
                                                             close=day_df[f"{pe_opt_symbol}_close"],
                                                             volume=day_df[f"{pe_opt_symbol}_volume"],
                                                             achored_time=self.strategy.start_time)
            day_df[f"{ce_opt_symbol}_vwap"] = day_df[f"{ce_opt_symbol}_vwap"] * (1 + self.strategy.buffer)
            day_df[f"{pe_opt_symbol}_vwap"] = day_df[f"{pe_opt_symbol}_vwap"] * (1 + self.strategy.buffer)
            day_df = day_df.reset_index()
            is_ce_position = False
            running_pnl_points = 0
            running_pnl_rupees = 0
            total_trades = 0
            quantity = self.strategy.position_size
            is_pe_position = False
            for idx, row in day_df.iterrows():
                curr_time = day_df.loc[idx, "date"].time()
                if curr_time >= self.strategy.end_time:
                    if is_ce_position:
                        buy_price = day_df.loc[idx, f"{ce_opt_symbol}_close"]
                        tradebook["datetime"].append(day_df.loc[idx, "date"])
                        tradebook["price"].append(buy_price)
                        tradebook["side"].append(1)
                        tradebook["traded_quantity"].append(quantity)
                        running_pnl_points += buy_price * 1
                        running_pnl_rupees += buy_price * 1 * quantity
                        self.backtest_logger.logger.info(
                            f"{curr_time} Exit: {ce_opt_symbol} End of trade px@{day_df.loc[idx, f'{ce_opt_symbol}_close']} qty:{quantity} PNL: {running_pnl_rupees * -1}"
                        )
                        total_trades += 1

                    if is_pe_position:
                        buy_price = day_df.loc[idx, f"{pe_opt_symbol}_close"]
                        tradebook["datetime"].append(day_df.loc[idx, "date"])
                        tradebook["price"].append(buy_price)
                        tradebook["side"].append(1)
                        tradebook["traded_quantity"].append(quantity)
                        running_pnl_points += buy_price * 1
                        running_pnl_rupees += buy_price * 1 * quantity
                        self.backtest_logger.logger.info(
                            f"{curr_time} Exit: {pe_opt_symbol} End of trade px@{buy_price} qty:{quantity} PNL: {running_pnl_rupees * -1}"
                        )
                        total_trades += 1
                    break
                if curr_time >= self.strategy.start_time and not is_ce_position:
                    if day_df.loc[idx, f"{ce_opt_symbol}_close"] < day_df.loc[idx, f"{ce_opt_symbol}_vwap"]:
                        curr_vwap = day_df.loc[idx, f"{ce_opt_symbol}_vwap"]
                        short_price = day_df.loc[idx, f"{ce_opt_symbol}_close"]
                        tradebook["datetime"].append(day_df.loc[idx, "date"])
                        tradebook["price"].append(short_price)
                        tradebook["side"].append(-1)
                        tradebook["traded_quantity"].append(quantity)
                        running_pnl_points += short_price * -1
                        running_pnl_rupees += short_price * -1 * quantity
                        is_ce_position = True
                        self.backtest_logger.logger.info(
                            f"{curr_time} Entry : {ce_opt_symbol} px@{short_price}, vwap:{curr_vwap} qty:{quantity}"
                        )

                if curr_time >= self.strategy.start_time and not is_pe_position:
                    if day_df.loc[idx, f"{pe_opt_symbol}_close"] < day_df.loc[idx, f"{pe_opt_symbol}_vwap"]:
                        curr_vwap = day_df.loc[idx, f"{pe_opt_symbol}_vwap"]
                        short_price = day_df.loc[idx, f"{pe_opt_symbol}_close"]
                        tradebook["datetime"].append(day_df.loc[idx, "date"])
                        tradebook["price"].append(short_price)
                        tradebook["side"].append(-1)
                        tradebook["traded_quantity"].append(quantity)
                        running_pnl_points += short_price * -1
                        running_pnl_rupees += short_price * -1 * quantity
                        is_pe_position = True
                        self.backtest_logger.logger.info(
                            f"{curr_time} Entry : {pe_opt_symbol} px@{short_price}, vwap:{curr_vwap} qty:{quantity}"
                        )

                if is_ce_position:
                    # If close > vwap, exit
                    if day_df.loc[idx, f"{ce_opt_symbol}_close"] > day_df.loc[idx, f"{ce_opt_symbol}_vwap"]:
                        curr_vwap = day_df.loc[idx, f"{ce_opt_symbol}_vwap"]
                        buy_price = day_df.loc[idx, f"{ce_opt_symbol}_close"]
                        tradebook["datetime"].append(day_df.loc[idx, "date"])
                        tradebook["price"].append(buy_price)
                        tradebook["side"].append(1)
                        tradebook["traded_quantity"].append(quantity)
                        running_pnl_points += buy_price * 1
                        running_pnl_rupees += buy_price * 1 * quantity
                        is_ce_position = False
                        self.backtest_logger.logger.info(
                            f"{curr_time} Exit : {ce_opt_symbol} px@{buy_price}, vwap: {curr_vwap}, qty: {quantity} PNL: {running_pnl_rupees * -1}")
                        quantity = self.strategy.position_size

                if is_pe_position:
                    # If close > vwap, exit
                    if day_df.loc[idx, f"{pe_opt_symbol}_close"] > day_df.loc[idx, f"{pe_opt_symbol}_vwap"]:
                        curr_vwap = day_df.loc[idx, f"{pe_opt_symbol}_vwap"]
                        buy_price = day_df.loc[idx, f"{pe_opt_symbol}_close"]
                        tradebook["datetime"].append(day_df.loc[idx, "date"])
                        tradebook["price"].append(buy_price)
                        tradebook["side"].append(1)
                        tradebook["traded_quantity"].append(quantity)
                        running_pnl_points += buy_price * 1
                        running_pnl_rupees += buy_price * 1 * quantity
                        is_pe_position = False
                        self.backtest_logger.logger.info(
                            f"{curr_time} Exit : {pe_opt_symbol} px@{buy_price}, vwap: {curr_vwap}, qty: {quantity} PNL: {running_pnl_rupees * -1}")
                        quantity = self.strategy.position_size

                # Check MTM SL HIT or NOT
                if (is_ce_position or is_pe_position) and self.strategy.stoploss_mtm_rupees:
                    curr_ce_close_px = day_df.loc[idx, f"{ce_opt_symbol}_close"]
                    curr_pe_close_px = day_df.loc[idx, f"{pe_opt_symbol}_close"]
                    curr_pnl = running_pnl_rupees
                    if is_ce_position:
                        curr_pnl += (curr_ce_close_px * 1 * quantity)
                    if is_pe_position:
                        curr_pnl += (curr_pe_close_px * 1 * quantity)
                    curr_pnl *= -1
                    if curr_pnl <= self.strategy.stoploss_mtm_rupees:
                        if is_ce_position:
                            self.backtest_logger.logger.info(
                                f"{curr_time}: MTM RUPEES SL HIT: {ce_opt_symbol} px@{curr_ce_close_px}, qty: {quantity} C.PNL: {curr_pnl}")
                            tradebook["datetime"].append(day_df.loc[idx, "date"])
                            tradebook["price"].append(curr_ce_close_px)
                            tradebook["side"].append(1)
                            tradebook["traded_quantity"].append(quantity)
                            running_pnl_points += curr_ce_close_px * 1
                            running_pnl_rupees += curr_ce_close_px * 1 * quantity
                        if is_pe_position:
                            tradebook["datetime"].append(day_df.loc[idx, "date"])
                            tradebook["price"].append(curr_pe_close_px)
                            tradebook["side"].append(1)
                            tradebook["traded_quantity"].append(quantity)
                            running_pnl_points += curr_pe_close_px * 1
                            running_pnl_rupees += curr_pe_close_px * 1 * quantity
                            self.backtest_logger.logger.info(
                                f"{curr_time} MTM RUPEES SL HIT: {pe_opt_symbol} px@{curr_pe_close_px} qty:{quantity} PNL: {running_pnl_rupees * -1}"
                            )
                        break

        tradebook["unable_to_trade_days"] = unable_to_trade_days
        save_tradebook(tradebook, self.strategy.strat_name)

    def backtest_atm_combined_tsl(self):
        df = self.data.resampled_historical_df
        df["day"] = df["date"].dt.date
        tradebook = {'datetime': [], 'price': [], 'side': [], 'traded_quantity': [],
                     "is_intraday": self.strategy.is_intraday,
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
            opt_symbols = utils.generate_opt_symbols_from_strike(self.strategy.instrument, entry_atm, expiry_comp,
                                                                 "ATM")
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

            day_df = day_df.reset_index()
            is_position = False
            running_pnl_points = 0
            running_pnl_rupees = 0
            total_trades = 0
            quantity = self.strategy.position_size
            is_half_squared_off = False
            stoploss = None
            prev_close_px = utils.INT_MAX
            is_sl_hit = False
            start_time = self.strategy.start_time
            end_time = self.strategy.end_time
            for idx, row in day_df.iterrows():
                curr_time = day_df.loc[idx, "date"].time()
                curr_close_px = day_df.loc[idx, 'combined_premium_close']
                if curr_time >= end_time:
                    if is_position:
                        self.backtest_logger.logger.info(
                            f"{curr_time} Exit: End of trade px@{curr_close_px} qty:{quantity}"
                        )
                        tradebook["datetime"].append(day_df.loc[idx, "date"])
                        tradebook["price"].append(curr_close_px)
                        tradebook["side"].append(1)
                        tradebook["traded_quantity"].append(quantity)
                        total_trades += 1
                    break
                if curr_time >= start_time and not is_position:
                    tradebook["datetime"].append(day_df.loc[idx, "date"])
                    tradebook["price"].append(curr_close_px)
                    tradebook["side"].append(-1)
                    tradebook["traded_quantity"].append(quantity)
                    running_pnl_points += curr_close_px * -1
                    running_pnl_rupees += curr_close_px * -1 * quantity
                    stoploss = curr_close_px * (1 + self.strategy.stoploss_pct)
                    is_position = True
                    self.backtest_logger.logger.info(
                        f"{curr_time} Entry :  px@{curr_close_px}, qty:{quantity} SL: {stoploss}"
                    )

                if is_position:
                    if curr_close_px > stoploss:
                        tradebook["datetime"].append(day_df.loc[idx, "date"])
                        tradebook["price"].append(stoploss)
                        tradebook["side"].append(1)
                        tradebook["traded_quantity"].append(quantity)
                        running_pnl_points += stoploss * 1
                        running_pnl_rupees += stoploss * 1 * quantity
                        self.backtest_logger.logger.info(
                            f"{curr_time} Exit SL HIT: px@{stoploss}, qty: {quantity} PNL: {running_pnl_rupees * -1}")
                        quantity = self.strategy.position_size
                        is_position = False
                        prev_close_px = utils.INT_MAX
                        stoploss = None
                        delta = timedelta(minutes=3)
                        start_time = (datetime.combine(date.today(), curr_time) + delta).time()
                        continue

                    if is_position and self.strategy.is_trail_sl and prev_close_px != utils.INT_MAX:
                        diff = prev_close_px - curr_close_px
                        if diff >= self.strategy.tsl[0]:
                            stoploss -= diff
                            self.backtest_logger.logger.info(
                                f"{curr_time}: curr_close:{curr_close_px} SL TRAILED from {stoploss + diff} to {stoploss}")

                    if is_position:
                        # Check MTM in rupees stoploss HIT or not
                        if self.strategy.stoploss_mtm_rupees:
                            curr_pnl_rupees = (curr_close_px * 1 * quantity + running_pnl_rupees) * -1
                            if curr_pnl_rupees <= self.strategy.stoploss_mtm_rupees:
                                self.backtest_logger.logger.info(
                                    f"{curr_time}: MTM RUPEES SL HIT: CLOSEpx@{curr_close_px}, qty: {quantity} C.PNL: {curr_pnl_rupees}")
                                tradebook["datetime"].append(day_df.loc[idx, "date"])
                                tradebook["price"].append(curr_close_px)
                                tradebook["side"].append(1)
                                tradebook["traded_quantity"].append(quantity)

                                continue

                        # Check MTM reached or not
                        if self.strategy.close_half_on_mtm_rupees and not is_half_squared_off:
                            curr_pnl_rupees = (curr_close_px * 1 * quantity + running_pnl_rupees) * -1
                            if curr_pnl_rupees >= self.strategy.close_half_on_mtm_rupees:
                                self.backtest_logger.logger.info(
                                    f"{curr_time} Square off half Qty: px@{curr_close_px} qty: {quantity / 2}:"
                                    f" C.PNL: {curr_pnl_rupees}"
                                )
                                tradebook["datetime"].append(day_df.loc[idx, "date"])
                                tradebook["price"].append(curr_close_px)
                                tradebook["side"].append(1)
                                tradebook["traded_quantity"].append(quantity / 2)
                                is_half_squared_off = True
                                quantity = quantity / 2
                                running_pnl_points += curr_close_px * 1
                                running_pnl_rupees += curr_close_px * 1 * quantity

                prev_close_px = min(curr_close_px, prev_close_px)
        tradebook["unable_to_trade_days"] = unable_to_trade_days
        save_tradebook(tradebook, self.strategy.strat_name)

    def backtest_atm_straddle_rolling_tsl(self):
        df = self.data.resampled_historical_df
        df["day"] = df["date"].dt.date
        tradebook = {'datetime': [], 'price': [], 'side': [], 'traded_quantity': [],
                     "is_intraday": self.strategy.is_intraday,
                     "unable_to_trade_days": 0}
        day_groups = df.groupby("day")
        unable_to_trade_days = 0
        for day, day_df in day_groups:
            curr_week_expiry_dt = self.data.get_closet_expiry(self.strategy.instrument, day,
                                                              week_number=self.strategy.expiry_week)
            expiry_comp = self.data.get_expiry_comp_from_date(self.strategy.instrument, curr_week_expiry_dt)
            entry_close_price = day_df.loc[day_df["date"].dt.time == self.strategy.start_time, "close"].iloc[0]
            entry_atm = int(round(entry_close_price, -2))
            ce_opt_symbol = utils.get_opt_symbol(self.strategy.instrument, expiry_comp, entry_atm, "CE")
            pe_opt_symbol = utils.get_opt_symbol(self.strategy.instrument, expiry_comp, entry_atm, "PE")
            opt_symbols = [ce_opt_symbol, pe_opt_symbol]
            self.backtest_logger.logger.info(f"Start: {day}, atm: {entry_atm}")
            self.backtest_logger.logger.info(f"symbols: {opt_symbols}")
            is_symbol_missing, day_df = utils.get_merged_opt_symbol_df(day_df, opt_symbols, day,
                                                                       self.strategy.opt_timeframe)

            if is_symbol_missing:
                with open(os.path.join(os.getcwd(), self.strategy.strat_name, "missing_symbols.txt"), "a") as f:
                    f.write(f"{day}\n")
                unable_to_trade_days += 1
                continue
            day_df.set_index("date", inplace=True)
            day_df = day_df.reset_index()
            is_position = False
            running_pnl_points = 0
            running_pnl_rupees = 0
            total_trades = 0
            quantity = self.strategy.position_size
            ce_stoploss = None
            pe_stoploss = None
            MAXX = 1e18
            prev_ce_close_px = MAXX
            prev_pe_close_px = MAXX
            start_time = self.strategy.start_time
            end_time = self.strategy.end_time
            re_execute_count = 0
            is_symbol_missing_inside = False
            for idx, row in day_df.iterrows():
                ce_candle = CandleData(candle_series=day_df.loc[idx], symbol=ce_opt_symbol)
                pe_candle = CandleData(candle_series=day_df.loc[idx], symbol=pe_opt_symbol)
                if ce_candle.time < start_time:
                    continue
                if ce_candle.time >= end_time and is_position:
                    tradebook, running_pnl_points, running_pnl_rupees = utils.add_trade(tradebook, running_pnl_points,
                                                                                        running_pnl_rupees,
                                                                                        ce_candle.date, ce_candle.close,
                                                                                        BUY, quantity)
                    tradebook, running_pnl_points, running_pnl_rupees = utils.add_trade(tradebook, running_pnl_points,
                                                                                        running_pnl_rupees,
                                                                                        pe_candle.date, pe_candle.close,
                                                                                        BUY, quantity)
                    self.backtest_logger.logger.info(
                        f"{ce_candle.time} Exit: {ce_opt_symbol} End of trade px@{ce_candle.close} qty:{quantity} PNL: {running_pnl_rupees * -1}")
                    self.backtest_logger.logger.info(
                        f"{ce_candle.time} Exit: {pe_opt_symbol} End of trade px@{pe_candle.close} qty:{quantity} PNL: {running_pnl_rupees * -1}")
                    total_trades += 2
                    break
                if ce_candle.time >= start_time and not is_position:
                    tradebook, running_pnl_points, running_pnl_rupees = utils.add_trade(tradebook, running_pnl_points,
                                                                                        running_pnl_rupees,
                                                                                        ce_candle.date, ce_candle.close,
                                                                                        SELL, quantity)
                    tradebook, running_pnl_points, running_pnl_rupees = utils.add_trade(tradebook, running_pnl_points,
                                                                                        running_pnl_rupees,
                                                                                        pe_candle.date, pe_candle.close,
                                                                                        SELL, quantity)
                    self.backtest_logger.logger.info(
                        f"{ce_candle.time} Entry : {ce_opt_symbol} px@{ce_candle.close}, qty:{quantity}")
                    self.backtest_logger.logger.info(
                        f"{ce_candle.time} Entry : {pe_opt_symbol} px@{pe_candle.close}, qty:{quantity}")
                    is_position = True
                    ce_stoploss = ce_candle.close * (1 + self.strategy.stoploss_pct)
                    pe_stoploss = pe_candle.close * (1 + self.strategy.stoploss_pct)

                if is_position:
                    if ce_candle.close > ce_stoploss:
                        tradebook, running_pnl_points, running_pnl_rupees = utils.add_trade(tradebook,
                                                                                            running_pnl_points,
                                                                                            running_pnl_rupees,
                                                                                            ce_candle.date,
                                                                                            ce_candle.close,
                                                                                            BUY, quantity)
                        tradebook, running_pnl_points, running_pnl_rupees = utils.add_trade(tradebook,
                                                                                            running_pnl_points,
                                                                                            running_pnl_rupees,
                                                                                            ce_candle.date,
                                                                                            pe_candle.close, BUY,
                                                                                            quantity)
                        self.backtest_logger.logger.info(
                            f"{ce_candle.time} Exiting:[SL HIT] {ce_opt_symbol}: px@{ce_candle.close}, qty: {quantity}")
                        self.backtest_logger.logger.info(
                            f"{ce_candle.time} Exiting {pe_opt_symbol}: px@{pe_candle.close}, qty: {quantity} ")
                        is_position = False
                        prev_ce_close_px = MAXX
                        prev_pe_close_px = MAXX
                        ce_stoploss = None
                        pe_stoploss = None
                        ce_opt_symbol = utils.get_opt_symbol(self.strategy.instrument, expiry_comp, ce_candle.atm,
                                                             OPTION_TYPE_PE)
                        pe_opt_symbol = utils.get_opt_symbol(self.strategy.instrument, expiry_comp, pe_candle.atm,
                                                             OPTION_TYPE_CE)
                        opt_symbols = [ce_opt_symbol, pe_opt_symbol]
                        is_symbol_missing, day_df = utils.get_merged_opt_symbol_df(day_df, opt_symbols, day,
                                                                                   self.strategy.opt_timeframe)
                        if is_symbol_missing:
                            is_symbol_missing_inside = True
                            break
                        delta = timedelta(minutes=0)
                        start_time = (datetime.combine(date.today(), ce_candle.time) + delta).time()
                        re_execute_count += 1
                        if re_execute_count >= self.strategy.re_execute_count:
                            break
                        continue

                    if pe_candle.close > pe_stoploss:
                        tradebook, running_pnl_points, running_pnl_rupees = utils.add_trade(tradebook,
                                                                                            running_pnl_points,
                                                                                            running_pnl_rupees,
                                                                                            ce_candle.date,
                                                                                            pe_candle.close,
                                                                                            BUY, quantity)
                        tradebook, running_pnl_points, running_pnl_rupees = utils.add_trade(tradebook,
                                                                                            running_pnl_points,
                                                                                            running_pnl_rupees,
                                                                                            ce_candle.date,
                                                                                            ce_candle.close, BUY,
                                                                                            quantity)
                        self.backtest_logger.logger.info(
                            f"{ce_candle.time} Exiting:[SL HIT] {pe_opt_symbol}: px@{pe_candle.close}, qty: {quantity} ")
                        self.backtest_logger.logger.info(
                            f"{ce_candle.time} Exit {ce_opt_symbol}: px@{ce_candle.close}, qty: {quantity} ")
                        is_position = False
                        prev_ce_close_px = MAXX
                        prev_pe_close_px = MAXX
                        ce_stoploss = None
                        pe_stoploss = None
                        ce_opt_symbol = utils.get_opt_symbol(self.strategy.instrument, expiry_comp, ce_candle.atm,
                                                             OPTION_TYPE_PE)
                        pe_opt_symbol = utils.get_opt_symbol(self.strategy.instrument, expiry_comp, pe_candle.atm,
                                                             OPTION_TYPE_CE)
                        opt_symbols = [ce_opt_symbol, pe_opt_symbol]
                        is_symbol_missing, day_df = utils.get_merged_opt_symbol_df(day_df, opt_symbols, day,
                                                                                   self.strategy.opt_timeframe)
                        if is_symbol_missing:
                            is_symbol_missing_inside = True
                            break
                        delta = timedelta(minutes=0)
                        start_time = (datetime.combine(date.today(), ce_candle.time) + delta).time()
                        re_execute_count += 1
                        if re_execute_count >= self.strategy.re_execute_count:
                            break
                        continue

                    if is_position and self.strategy.is_trail_sl and prev_ce_close_px != MAXX:
                        diff = prev_ce_close_px - ce_candle.close
                        if diff >= self.strategy.tsl[0]:
                            ce_stoploss -= diff
                            self.backtest_logger.logger.info(
                                f"{ce_candle.time}: {ce_opt_symbol} curr_close:{ce_candle.close} prev_close: {prev_ce_close_px} SL TRAILED from {ce_stoploss + diff} to {ce_stoploss}")

                    if is_position and self.strategy.is_trail_sl and prev_pe_close_px != MAXX:
                        diff = prev_pe_close_px - pe_candle.close
                        if diff >= self.strategy.tsl[0]:
                            pe_stoploss -= diff
                            self.backtest_logger.logger.info(
                                f"{ce_candle.time}:{pe_opt_symbol} curr_close:{pe_candle.close} prv_close: {prev_pe_close_px} SL TRAILED from {pe_stoploss + diff} to {pe_stoploss}")

                prev_ce_close_px = min(prev_ce_close_px, ce_candle.close)
                prev_pe_close_px = min(prev_pe_close_px, pe_candle.close)

            if is_symbol_missing_inside:
                with open(os.path.join(os.getcwd(), self.strategy.strat_name, "missing_symbols.txt"), "a") as f:
                    f.write(f"{day}\n")
                unable_to_trade_days += 1
                utils.remove_trades(tradebook, day)

        tradebook["unable_to_trade_days"] = unable_to_trade_days
        save_tradebook(tradebook, self.strategy.strat_name)

    def backtest_atr_buying(self):
        df = self.data.resampled_historical_df
        daily_atr_df = resample_ohlc_df(df, "D")
        daily_atr_df["date"] = daily_atr_df["date"].dt.date
        daily_atr_df["atr"] = ta.atr(high=daily_atr_df["high"], low=daily_atr_df["low"],
                                     close=daily_atr_df["close"], length=14)
        daily_atr_df.set_index("date", inplace=True)
        df["day"] = df["date"].dt.date
        tradebook = {'datetime': [], 'price': [], 'side': [], 'traded_quantity': [],
                     "is_intraday": self.strategy.is_intraday, "unable_to_trade_days": 0}
        day_groups = df.groupby("day")
        unable_to_trade_days = 0
        prev_day = None
        for day, day_df in day_groups:
            curr_week_expiry_dt = self.data.get_closet_expiry(self.strategy.instrument, day,
                                                              week_number=self.strategy.expiry_week)
            expiry_comp = self.data.get_expiry_comp_from_date(self.strategy.instrument, curr_week_expiry_dt)
            if prev_day is None or daily_atr_df.loc[day].empty or daily_atr_df.loc[day].isnull().values.any():
                prev_day = day
                continue
            else:
                prev_day = day
                prev_14day_atr = daily_atr_df.loc[day, "atr"]
            day_df.set_index("date", inplace=True)
            day_df = day_df.reset_index()
            running_pnl_points = 0
            running_pnl_rupees = 0
            total_trades = 0
            quantity = self.strategy.position_size
            start_time = self.strategy.start_time
            end_time = self.strategy.end_time
            re_execute_count = 0
            spot_day_high = utils.INT_MIN
            spot_day_low = utils.INT_MAX
            is_ce_position = False
            is_pe_position = False
            for idx, row in day_df.iterrows():
                spot_candle = CandleData(row)
                if spot_day_high != utils.INT_MIN:
                    diff = spot_day_high - day_df.loc[idx, "close"]
                    atr_percentage = 1.1 * prev_14day_atr
                    if diff > atr_percentage and not is_pe_position:
                        ce_opt_symbol = utils.get_opt_symbol(self.strategy.instrument, expiry_comp,
                                                             spot_candle.atm, OPTION_TYPE_PE)
                        is_symbol_missing, day_df = utils.get_merged_opt_symbol_df(day_df, [ce_opt_symbol], day,
                                                                                   self.strategy.opt_timeframe)
                spot_day_high = max(spot_day_high, day_df.loc[idx, "high"])
                spot_day_low = min(spot_day_low, day_df.loc[idx, "low"])

        tradebook["unable_to_trade_days"] = unable_to_trade_days
        save_tradebook(tradebook, self.strategy.strat_name)
