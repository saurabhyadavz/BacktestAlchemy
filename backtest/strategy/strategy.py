from typing import Union


class Strategy:
    def __init__(self, is_intraday: bool, start_time: str, end_time: str, stop_loss: Union[int, float] = None,
                 target: Union[int, float] = None, per_trade_commission: float = 0, slippage: float = 0,
                 re_entry_count: int = 0, re_execute_count: int = 0):
        self.is_intraday = is_intraday
        self.start_time = start_time
        self.end_time = end_time
        self.stop_loss = stop_loss
        self.target = target
        self.per_trade_commission = per_trade_commission
        self.slippage = slippage
        self.re_entry_count = re_entry_count
        self.re_execute_count = re_execute_count
