import pandas as pd
from backtest.strategy.strategy import Strategy
from backtest.data.data import DataProducer


class OptBacktest:
    def __init__(self, strategy: Strategy, data: DataProducer):
        self.strategy = strategy
        self.data = data

    def backtest_simple_straddle(self):
        df = self.data.historical_df_with_exp_comp
        df["day"] = df["date"].dt.date
        df["Points"] = 0
        date_points = pd.Series(index=df['date'].dt.date.unique())
        day_groups = df.groupby("day")
        for day, day_df in day_groups:
            print(f"Running for {day}")
            entry_close_price = day_df.loc[day_df["date"].dt.time == self.strategy.start_time, "close"].iloc[0]
            atm = int(round(entry_close_price, -2))
            expiry_comp = day_df["expiry_comp"].iloc[0]
            option_symbols = []
            atm_ce_symbol = f"{self.data.instrument}{expiry_comp}{atm}CE"
            atm_pe_symbol = f"{self.data.instrument}{expiry_comp}{atm}PE"
            option_symbols.append(atm_ce_symbol)
            option_symbols.append(atm_pe_symbol)
            is_symbol_missing = False
            for opt_symbol in option_symbols:
                curr_date = day
                option_df = self.data.fetch_options_data_and_resample(opt_symbol, curr_date, curr_date,
                                                                      self.strategy.timeframe)
                if option_df.empty:
                    is_symbol_missing = True
                    with open(r"C:\Users\DeGenOne\degen-money-backtest\backtest\missing_symbols.txt", "a") as f:
                        f.write(f"{day} {opt_symbol}\n")
                else:
                    day_df = pd.merge(day_df, option_df, on='date')
            if is_symbol_missing:
                continue
            ce_stoploss_price, pe_stoploss_price = None, None
            ce_short_price, pe_short_price = None, None
            day_pnl = 0
            is_ce_leg_open, is_pe_leg_open = False, False
            is_position = False
            for idx, row in day_df.iterrows():
                curr_time = row["date"].time()
                curr_ce_price = row[f"{atm_ce_symbol}_close"]
                curr_pe_price = row[f"{atm_pe_symbol}_close"]
                if curr_time < self.strategy.start_time:
                    continue
                if curr_time >= self.strategy.start_time and not is_position:
                    ce_short_price = curr_ce_price
                    pe_short_price = curr_pe_price
                    is_ce_leg_open, is_pe_leg_open = True, True
                    is_position = True
                    ce_stoploss_price = (1 + self.strategy.stop_loss) * ce_short_price
                    pe_stoploss_price = (1 + self.strategy.stop_loss) * pe_short_price
                else:
                    if curr_time >= self.strategy.end_time:
                        if is_ce_leg_open:
                            day_pnl += (ce_short_price - curr_ce_price)
                            day_df.loc[idx, "Points"] += (ce_short_price - curr_ce_price)
                        if is_pe_leg_open:
                            day_pnl += (pe_short_price - curr_pe_price)
                            day_df.loc[idx, "Points"] += (pe_short_price - curr_pe_price)
                        break
                    if curr_ce_price >= ce_stoploss_price and is_ce_leg_open:
                        day_pnl += (ce_short_price - curr_ce_price)
                        day_df.loc[idx, "Points"] += (ce_short_price - curr_ce_price)
                        is_ce_leg_open = False
                        if is_pe_leg_open and self.strategy.move_sl_to_cost:
                            pe_stoploss_price = pe_short_price

                    if curr_pe_price >= pe_stoploss_price and is_pe_leg_open:
                        day_pnl += (pe_short_price - curr_pe_price)
                        day_df.loc[idx, "Points"] += (pe_short_price - curr_pe_price)
                        is_pe_leg_open = False
                        if is_ce_leg_open and self.strategy.move_sl_to_cost:
                            ce_stoploss_price = ce_short_price
            date_points.loc[day] = day_df['Points'].sum()
        return date_points
