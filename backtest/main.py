###################### adding project path to python path
import sys

sys.path.append(r"C:\Users\DeGenOne\degen-money-backtest")
#####################
import os
from backtest.data.data import DataProducer
from backtest.strategy.strategy import Strategy
from backtest.opt_backtest.opt_backtest import OptBacktest
from backtest.analyzers.analyzers import Analyzers, DegenPlotter


# def run_simple_straddle():
#     strategy = Strategy(start_date="2019-01-01", end_date="2019-12-30",
#                         instrument="NIFTY", is_intraday=True,
#                         start_time="9:25", end_time="15:00",
#                         stop_loss=0.2, move_sl_to_cost=True)
#     data = DataProducer(strategy.instrument, strategy.start_date, strategy.end_date, strategy.timeframe)
#     bt = OptBacktest(strategy, data)
#     points = bt.backtest_simple_straddle()
#     analyzer = Analyzers()
#     metrics = analyzer.get_metrics(points, strat="920 STRADDLE 20% SL MOVE TO COST")
#     print(metrics)


def run_positional_iron_condor():
    strategy = Strategy(start_date="2019-01-01", end_date="2019-12-30",
                        instrument="BANKNIFTY", is_intraday=False,
                        start_time="10:00", end_time="10:00",
                        re_execute_count=1, trading_days_before_expiry=1,
                        how_far_otm_hedge_point=800, how_far_otm_short_point=400)
    data = DataProducer(strategy.instrument, strategy.start_date, strategy.end_date, strategy.timeframe)
    bt = OptBacktest(strategy, data)
    points = bt.backtest_positional_iron_condor()
    degen_plotter = DegenPlotter(points, lot_size=25, strat_name="positional_iron_condor")
    degen_plotter.plot_all()
    analyzer = Analyzers()
    metrics = analyzer.get_metrics(points, strat="Positional Iron Condor")
    metrics.to_csv(os.path.join(os.getcwd(), "metrices.csv"))


if __name__ == "__main__":
    run_positional_iron_condor()
