# adding project path to python path
import sys

sys.path.append(r"C:\Users\DeGenOne\degen-money-backtest")

from backtest.data.data import DataProducer
from backtest.strategy.strategy import Strategy

if __name__ == "__main__":
    data = DataProducer("BANKNIFTY", "2022-01-01", "2022-01-01", "60min")
    strat = Strategy(is_intraday=False, start_time="9:25", end_time="15:00")

