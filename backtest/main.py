# Adding project path to PYTHONPATH
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backtest.data.data import DataProducer
from backtest.strategy.strategy import Strategy
from backtest.opt_backtest.opt_backtest import OptBacktest
from backtest.analyzers.analyzers import Analyzers


def run_combined_premium(start_date: str, end_date: str, strat_name: str, buffer: float, run_backtest: bool = True):
    if run_backtest:
        print(f"Running {strat_name}")
        strategy = Strategy(strat_name=strat_name, start_date=start_date, end_date=end_date,
                            instrument="BANKNIFTY", capital=1000000, lots=1, is_intraday=True,
                            start_time="9:20", end_time="15:00", timeframe="5min", opt_timeframe="5min", expiry_week=0,
                            stoploss_mtm_rupees=10000, buffer=buffer)
        data = DataProducer(strategy.instrument, strategy.start_date, strategy.end_date, strategy.timeframe)
        bt = OptBacktest(strategy, data)
        bt.backtest_combined_premium_vwap()
    Analyzers(capital=1000000, instrument="BANKNIFTY", lots=1, start_date=start_date, end_date=end_date,
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


def run_itm_vwap(start_date: str, end_date: str, strat_name: str, run_backtest: bool = True):
    if run_backtest:
        print(f"Running {strat_name}")
        strategy = Strategy(strat_name=strat_name, start_date=start_date, end_date=end_date,
                            instrument="BANKNIFTY", capital=200000, lots=1, is_intraday=True,
                            start_time="9:45", end_time="14:45", timeframe="30min", opt_timeframe="30min",
                            stoploss_mtm_rupees=2000, expiry_week=0)
        data = DataProducer(strategy.instrument, strategy.start_date, strategy.end_date, strategy.timeframe)
        bt = OptBacktest(strategy, data)
        bt.backtest_itm_vwap()
    Analyzers(capital=200000, instrument="BANKNIFTY", lots=1, start_date=start_date, end_date=end_date,
              strat_name=strat_name, slippage=0.005)


if __name__ == "__main__":
    start_date = "2021-01-01"
    end_date = "2021-12-31"
    strat_name = "itm5_vwap"
    run_itm_vwap(start_date=start_date, end_date=end_date, strat_name=f"{strat_name}_TF30MIN_500ITM", run_backtest=True)
