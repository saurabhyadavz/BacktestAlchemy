# adding project path to python path
import sys

sys.path.append(r"C:\Users\DeGenOne\degen-money-backtest")

from backtest.data.data import DataProducer

if __name__ == "__main__":
    data = DataProducer("BANKNIFTY", "2022-01-01", "2022-01-01", "60min")
    print(data.resampled_historical_df)
