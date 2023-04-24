from backtest.strategy.strategy import Strategy


class SpotBacktest(Strategy):
    def __init__(self, is_intraday: bool, start_time: str, end_time: str):
        super().__init__(is_intraday, start_time, end_time)
