import numpy as np
import pandas as pd
import os
from backtest.degen_logger.degen_logger import DegenLogger
from backtest.strategy.strategy import Strategy
from backtest.data.data import DataProducer
from backtest.utils import utils


class OptBacktest:
    def __init__(self, strategy: Strategy, data: DataProducer):
        self.strategy = strategy
        self.data = data
        self.backtest_logger = DegenLogger(os.path.join(os.getcwd(), "backtest.log"))

    # def backtest_simple_straddle(self):
    #     df = self.data.historical_df_with_exp_comp
    #     df["day"] = df["date"].dt.date
    #     df["Points"] = 0
    #     date_points = pd.Series(index=df['date'].dt.date.unique(), dtype=object)
    #     day_groups = df.groupby("day")
    #     for day, day_df in day_groups:
    #         entry_close_price = day_df.loc[day_df["date"].dt.time == self.strategy.start_time, "close"].iloc[0]
    #         atm = int(round(entry_close_price, -2))
    #         expiry_comp = day_df["expiry_comp"].iloc[0]
    #         option_symbols = []
    #         atm_ce_symbol = f"{self.data.instrument}{expiry_comp}{atm}CE"
    #         atm_pe_symbol = f"{self.data.instrument}{expiry_comp}{atm}PE"
    #         option_symbols.append(atm_ce_symbol)
    #         option_symbols.append(atm_pe_symbol)
    #         is_symbol_missing = False
    #         for opt_symbol in option_symbols:
    #             curr_date = day
    #             option_df = self.data.fetch_options_data_and_resample(opt_symbol, curr_date, curr_date,
    #                                                                   self.strategy.timeframe)
    #             if option_df.empty:
    #                 is_symbol_missing = True
    #                 with open(r"C:\Users\DeGenOne\degen-money-backtest\backtest\missing_symbols.txt", "a") as f:
    #                     f.write(f"{day} {opt_symbol}\n")
    #             else:
    #                 day_df = pd.merge(day_df, option_df, on='date')
    #         if is_symbol_missing:
    #             continue
    #         ce_stoploss_price, pe_stoploss_price = None, None
    #         ce_short_price, pe_short_price = None, None
    #         day_pnl = 0
    #         is_ce_leg_open, is_pe_leg_open = False, False
    #         is_position = False
    #         for idx, row in day_df.iterrows():
    #             curr_time = row["date"].time()
    #             curr_ce_price = row[f"{atm_ce_symbol}_close"]
    #             curr_pe_price = row[f"{atm_pe_symbol}_close"]
    #             if curr_time < self.strategy.start_time:
    #                 continue
    #             if curr_time >= self.strategy.start_time and not is_position:
    #                 ce_short_price = curr_ce_price
    #                 pe_short_price = curr_pe_price
    #                 is_ce_leg_open, is_pe_leg_open = True, True
    #                 is_position = True
    #                 ce_stoploss_price = (1 + self.strategy.stop_loss) * ce_short_price
    #                 pe_stoploss_price = (1 + self.strategy.stop_loss) * pe_short_price
    #             else:
    #                 if curr_time >= self.strategy.end_time:
    #                     if is_ce_leg_open:
    #                         day_pnl += (ce_short_price - curr_ce_price)
    #                         day_df.loc[idx, "Points"] += (ce_short_price - curr_ce_price)
    #                     if is_pe_leg_open:
    #                         day_pnl += (pe_short_price - curr_pe_price)
    #                         day_df.loc[idx, "Points"] += (pe_short_price - curr_pe_price)
    #                     break
    #                 if curr_ce_price >= ce_stoploss_price and is_ce_leg_open:
    #                     day_pnl += (ce_short_price - curr_ce_price)
    #                     day_df.loc[idx, "Points"] += (ce_short_price - curr_ce_price)
    #                     is_ce_leg_open = False
    #                     if is_pe_leg_open and self.strategy.move_sl_to_cost:
    #                         pe_stoploss_price = pe_short_price
    #
    #                 if curr_pe_price >= pe_stoploss_price and is_pe_leg_open:
    #                     day_pnl += (pe_short_price - curr_pe_price)
    #                     day_df.loc[idx, "Points"] += (pe_short_price - curr_pe_price)
    #                     is_pe_leg_open = False
    #                     if is_ce_leg_open and self.strategy.move_sl_to_cost:
    #                         ce_stoploss_price = ce_short_price
    #         date_points.loc[day] = day_df['Points'].sum()
    #     return date_points

    def backtest_positional_iron_condor(self):
        df = self.data.resampled_historical_df
        df["day"] = df["date"].dt.date
        df["Points"] = 0
        track_pnl = pd.Series(np.nan, index=df['date'])
        day_groups = df.groupby("day")
        is_position = False
        iron_condor_dict = {}
        expiry_comp = None
        re_execute = False
        re_execute_count = 0
        is_condition_satisfied = False
        entry_date = None
        exit_date = None
        for day, day_df in day_groups:
            curr_week_expiry_dt = self.data.get_closet_expiry(day, week_number=0)
            next_week_expiry_dt = self.data.get_closet_expiry(day, week_number=1)
            entry_n_days_before_expiry = utils.get_n_days_before_date(self.strategy.trading_days_before_expiry,
                                                                      self.data.trading_days, curr_week_expiry_dt)
            exit_n_days_before_expiry = utils.get_n_days_before_date(self.strategy.trading_days_before_expiry,
                                                                     self.data.trading_days, next_week_expiry_dt)
            if entry_n_days_before_expiry == day and not is_position:
                entry_date = day
                re_execute_count = 0
                exit_date = exit_n_days_before_expiry
                is_condition_satisfied = True
                entry_close_price = day_df.loc[day_df["date"].dt.time == self.strategy.start_time, "close"].iloc[0]
                entry_atm_strike = int(round(entry_close_price, -2))
                expiry_comp = self.data.get_expiry_comp_from_date(next_week_expiry_dt)
                iron_condor_dict = utils.generate_iron_condor_strikes_and_symbols(
                    self.strategy.instrument, entry_atm_strike,
                    self.strategy.how_far_otm_short_point,
                    self.strategy.how_far_otm_hedge_point, expiry_comp
                )

            if not is_condition_satisfied and not is_position:
                continue
            if is_position:
                is_symbol_missing, day_df = utils.get_merged_opt_symbol_df(day_df, iron_condor_dict, day,
                                                                           self.strategy.timeframe)
                # if any trading symbol is missing then we skip trading
                if is_symbol_missing:
                    continue

            day_df = day_df.reset_index(drop=True)
            for idx, row in day_df.iterrows():
                curr_time = day_df.loc[idx, "date"].time()
                curr_atm_strike = int(round(day_df.loc[idx, "close"], -2))

                if ((
                        curr_time >= self.strategy.start_time and not is_position and re_execute_count < self.strategy.re_execute_count)
                        or (re_execute and not is_position)):
                    self.backtest_logger.logger.info(f"Taking Entry at: {curr_time} on {day}")
                    is_symbol_missing, day_df = utils.get_merged_opt_symbol_df(day_df, iron_condor_dict, day,
                                                                               self.strategy.timeframe)
                    if is_symbol_missing:
                        break

                    ce_short_entry_price = day_df.loc[idx, f"{iron_condor_dict['ce_short_opt_symbol']}_close"]
                    pe_short_entry_price = day_df.loc[idx, f"{iron_condor_dict['pe_short_opt_symbol']}_close"]
                    ce_hedge_entry_price = day_df.loc[idx, f"{iron_condor_dict['ce_hedge_opt_symbol']}_close"]
                    pe_hedge_entry_price = day_df.loc[idx, f"{iron_condor_dict['pe_hedge_opt_symbol']}_close"]
                    self.backtest_logger.logger.info(f"Entry in {iron_condor_dict['ce_short_opt_symbol']}"
                                                     f" at {ce_short_entry_price}")
                    self.backtest_logger.logger.info(f"Entry in {iron_condor_dict['pe_short_opt_symbol']} "
                                                     f"at {pe_short_entry_price}")
                    self.backtest_logger.logger.info(f"Entry in {iron_condor_dict['ce_hedge_opt_symbol']} "
                                                     f"at {ce_hedge_entry_price}")
                    self.backtest_logger.logger.info(f"Entry in {iron_condor_dict['pe_hedge_opt_symbol']} "
                                                     f"at {pe_hedge_entry_price}")
                    total_entry_price = (-1 * ce_short_entry_price + -1 * pe_short_entry_price +
                                         1 * ce_hedge_entry_price + 1 * pe_hedge_entry_price)
                    track_pnl.loc[day_df.loc[idx, "date"]] = total_entry_price
                    is_position = True
                    if re_execute:
                        re_execute = False
                        re_execute_count += 1
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
                        track_pnl.loc[day_df.loc[idx, "date"]] = total_exit_price
                        is_position = False
                        self.backtest_logger.logger.info(f"Exiting at {curr_time} on {day}")
                        self.backtest_logger.logger.info(f"Exiting in {iron_condor_dict['ce_short_opt_symbol']}"
                                                         f" at {ce_short_exit_price}")
                        self.backtest_logger.logger.info(f"Exiting in {iron_condor_dict['pe_short_opt_symbol']} "
                                                         f"at {pe_short_exit_price}")
                        self.backtest_logger.logger.info(f"Exiting in {iron_condor_dict['ce_hedge_opt_symbol']} "
                                                         f"at {ce_hedge_exit_price}")
                        self.backtest_logger.logger.info(f"Exiting in {iron_condor_dict['pe_hedge_opt_symbol']} "
                                                         f"at {pe_hedge_exit_price}")

                        # re-execute only if:
                        # 1. current day is not exit day and curr time is greater than exit time
                        # 2. re-execute count limit is not reached
                        if (self.strategy.re_execute_count and
                                self.strategy.re_execute_count != re_execute_count and
                                not (day == exit_date and curr_time >= self.strategy.end_time)):
                            self.backtest_logger.logger.info(f"Re-Execute at next minute of {curr_time}")
                            re_execute = True

                if re_execute or (not is_position and day == exit_date):
                    if day == exit_date:
                        expiry_comp = self.data.get_expiry_comp_from_date(next_week_expiry_dt)
                        exit_date = exit_n_days_before_expiry
                        re_execute_count = 0
                    iron_condor_dict = utils.generate_iron_condor_strikes_and_symbols(
                        self.strategy.instrument, curr_atm_strike,
                        self.strategy.how_far_otm_short_point,
                        self.strategy.how_far_otm_hedge_point, expiry_comp
                    )

        return track_pnl
