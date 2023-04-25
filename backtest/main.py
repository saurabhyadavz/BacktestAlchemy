# adding project path to python path
import sys

sys.path.append(r"C:\Users\DeGenOne\degen-money-backtest")

from backtest.data.data import DataProducer
from backtest.strategy.strategy import Strategy
from backtest.opt_backtest.opt_backtest import OptBacktest
from backtest.analyzers.analyzers import Analyzers

if __name__ == "__main__":
    strategy = Strategy(start_date="2019-01-01", end_date="2019-12-30",
                        instrument="NIFTY", is_intraday=True,
                        start_time="9:25", end_time="15:00",
                        stop_loss=0.2, move_sl_to_cost=True)
    data = DataProducer(strategy.instrument, strategy.start_date, strategy.end_date, strategy.timeframe)
    bt = OptBacktest(strategy, data)
    points = bt.backtest_simple_straddle()
    analyzer = Analyzers()
    metrics = analyzer.get_metrics(points, strat="920 STRADDLE 20% SL MOVE TO COST")
    print(metrics)
