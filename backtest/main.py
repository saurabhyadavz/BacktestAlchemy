# Adding project path to PYTHONPATH
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import concurrent.futures
import threading
from backtest.data.data import DataProducer
from backtest.strategy.strategy import Strategy
from backtest.opt_backtest.opt_backtest import OptBacktest
from backtest.analyzers.analyzers import Analyzers


def run_combined_premium(start_date: str, end_date: str, strat_name: str, buffer: float, run_backtest: bool = True):
    if run_backtest:
        print(f"Running {strat_name}")
        strategy = Strategy(strat_name=strat_name, start_date=start_date, end_date=end_date,
                            instrument="BANKNIFTY", capital=2000000, lots=2, is_intraday=True,
                            start_time="9:20", end_time="15:10", timeframe="1min", opt_timeframe="5min", expiry_week=0,
                            stoploss_mtm_rupees=14000, buffer=buffer, close_half_on_mtm_rupees=10000)
        data = DataProducer(strategy.instrument, strategy.start_date, strategy.end_date, strategy.timeframe)
        bt = OptBacktest(strategy, data)
        bt.backtest_combined_premium_vwap()
    Analyzers(capital=2000000, instrument="BANKNIFTY", lots=2, start_date=start_date, end_date=end_date,
              strat_name=strat_name, slippage=0.005)


def run_simple_straddle(start_date: str, end_date: str, strat_name: str, run_backtest: bool = True):
    if run_backtest:
        strategy = Strategy(strat_name=strat_name, start_date=start_date, end_date=end_date,
                            instrument="BANKNIFTY", capital=1000000, lots=1, is_intraday=True,
                            start_time="9:20", end_time="15:10", timeframe="1min", expiry_week=0,
                            stoploss_pct=0.2, move_sl_to_cost=True)
        data = DataProducer(strategy.instrument, strategy.start_date, strategy.end_date, strategy.timeframe)
        bt = OptBacktest(strategy, data)
        bt.backtest_simple_straddle()
    Analyzers(capital=180000, instrument="BANKNIFTY", lots=1, start_date=start_date, end_date=end_date,
              strat_name=strat_name, slippage=0.01)


if __name__ == "__main__":
    start_date = "2019-01-01"
    end_date = "2022-12-31"
    strat_name = "combined_straddle_strangle_vwap"
    run_combined_premium(start_date, end_date, f"{strat_name}_CLOSE_HALF_ON10K", 1 / 100, True)
