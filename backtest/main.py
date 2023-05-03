# Adding project path to PYTHONPATH
import sys
import os

import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import concurrent.futures
from datetime import date, timedelta
from backtest.data.data import DataProducer
from backtest.strategy.strategy import Strategy
from backtest.opt_backtest.opt_backtest import OptBacktest
from backtest.analyzers.analyzers import Analyzers, DegenPlotter


def run_simple_straddle():
    strategy = Strategy(start_date="2018-01-01", end_date="2019-12-30",
                        instrument="NIFTY", is_intraday=True,
                        start_time="9:25", end_time="15:00",
                        stop_loss=0.2, move_sl_to_cost=True)
    data = DataProducer(strategy.instrument, strategy.start_date, strategy.end_date, strategy.timeframe)
    bt = OptBacktest(strategy, data)
    points = bt.backtest_simple_straddle()
    points.to_csv("points_2020_2022.csv")
    degen_plotter = DegenPlotter(points, lot_size=25, strat_name="Straddle 925")
    degen_plotter.plot_all()
    analyzer = Analyzers()
    metrics = analyzer.get_metrics(points, strat="Straddle 925")
    metrics.to_csv(os.path.join(os.getcwd(), "metrices.csv"))


def run_positional_iron_condor():
    strategy = Strategy(start_date="2020-01-01", end_date="2020-03-30",
                        instrument="BANKNIFTY", is_intraday=False,
                        start_time="10:00", end_time="10:00",
                        re_execute_count=1, trading_days_before_expiry=1,
                        how_far_otm_hedge_point=800, how_far_otm_short_point=400)
    data = DataProducer(strategy.instrument, strategy.start_date, strategy.end_date, strategy.timeframe)
    bt = OptBacktest(strategy, data)
    points = bt.backtest_positional_iron_condor()
    points.to_csv("points_2020_2022.csv")

    degen_plotter = DegenPlotter(points, lot_size=25, strat_name="Positional Iron Condor")
    degen_plotter.plot_all()
    analyzer = Analyzers()
    metrics = analyzer.get_metrics(points, strat="Positional Iron Condor")
    metrics.to_csv(os.path.join(os.getcwd(), "metrices.csv"))


def run_combined_premium(start_date: str, end_date: str):
    strategy = Strategy(start_date=start_date, end_date=end_date,
                        instrument="BANKNIFTY", is_intraday=True,
                        start_time="9:20", end_time="15:10", timeframe="5min", expiry_week=0,
                        stop_loss=-400, slippage=0.01)
    data = DataProducer(strategy.instrument, strategy.start_date, strategy.end_date, strategy.timeframe)
    bt = OptBacktest(strategy, data)
    unable_to_trade_days, points, trades = bt.backtest_combined_premium_vwap()
    points.to_csv("points.csv")
    trades.to_csv("trades.csv")
    degen_plotter = DegenPlotter(points, lot_size=25, strat_name="Combined Premium VWAP")
    degen_plotter.plot_all()
    # points = pd.read_csv("points.csv")
    analyzer = Analyzers(capital=1000000, instrument=strategy.instrument, lots=3, start_date=strategy.start_date,
                         end_date=strategy.end_date, strat_name="Combined Premium VWAP")
    metrics = analyzer.get_matrices(points, unable_to_trade_days=unable_to_trade_days)
    metrics.to_csv(os.path.join(os.getcwd(), "metrices.csv"), index_label='Test Start Date')


if __name__ == "__main__":
    run_combined_premium("2019-01-01", "2019-12-31")
    # total_unable_to_trade_days = 0
    # points_list = []
    # import time
    # start_time = time.time()
    #
    # with concurrent.futures.ThreadPoolExecutor(max_workers=7) as executor:
    #     future_results = []
    #     for year in range(2016, 2023):
    #         start_date = date(year, 1, 1).strftime('%Y-%m-%d')
    #         end_date = date(year, 12, 31).strftime('%Y-%m-%d')
    #         print(start_date, end_date)
    #         future = executor.submit(run_combined_premium, start_date, end_date)
    #         future_results.append(future)
    #
    #     for future in concurrent.futures.as_completed(future_results):
    #         unable_to_trade_days, points = future.result()
    #         randomm = random.randint(0, 100)
    #         points.to_csv(f"{randomm}_points.csv")
    #         total_unable_to_trade_days += unable_to_trade_days
    #         points_list.append(points)
    #
    # df = pd.concat(points_list)
    #
    # # df = df.sort_values(by='date')
    # df.to_csv("points_2020_2022.csv")
    #
    # end_time = time.time()
    # total_time = end_time - start_time
    #
    # print(f'The program took {total_time:.2f} seconds to run.')

