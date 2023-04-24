# adding project path to python path
import sys

sys.path.append("/Users/saurabh/degen-money-backtest")

from backtest.data.data import DataProducer
from backtest.strategy.strategy import Strategy
from backtest.opt_backtest.opt_backtest import OptBacktest
from backtest.analyzers.analyzers import Analyzers
if __name__ == "__main__":
    data = DataProducer("BANKNIFTY", "2020-01-01", "2020-02-20", "1min")
    strategy = Strategy(instrument="BANKNIFTY", is_intraday=False, start_time="9:25", end_time="15:00",
                        stop_loss=0.2, move_sl_to_cost=True)
    bt = OptBacktest(strategy, data)
    points = bt.backtest_simple_straddle()
    analyzer = Analyzers()
    metrics = analyzer.get_metrics(points, strat="920 STRADDLE 20% SL MOVE TO COST")
    print(metrics)
