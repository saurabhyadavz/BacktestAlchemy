# Adding project path to PYTHONPATH
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtest.data.data import DataProducer
from backtest.strategy.strategy import Strategy
from backtest.opt_backtest.opt_backtest import OptBacktest
from backtest.analyzers.analyzers import Analyzers


def run_combined_premium(start_date: str, end_date: str, strat_name: str, run_backtest: bool = True):
    if run_backtest:
        strategy = Strategy(strat_name=strat_name, start_date=start_date, end_date=end_date,
                            instrument="BANKNIFTY", capital=1000000, lots=3, is_intraday=True,
                            start_time="9:20", end_time="15:10", timeframe="5min", expiry_week=0,
                            stoploss_mtm_rupees=200000)
        data = DataProducer(strategy.instrument, strategy.start_date, strategy.end_date, strategy.timeframe)
        bt = OptBacktest(strategy, data)
        bt.backtest_combined_premium_vwap()
    Analyzers(capital=1000000, instrument="BANKNIFTY", lots=1, start_date=start_date, end_date=end_date,
              strat_name=strat_name, slippage=0.025)


if __name__ == "__main__":
    start_date = "2019-01-01"
    end_date = "2022-12-31"
    strat_name = "Combined Premium VWAP"
    run_combined_premium(start_date, end_date, strat_name, run_backtest=False)

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
