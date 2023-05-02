from typing import Union
from backtest.utils import utils


class Strategy:
    def __init__(self, start_date: str, end_date: str, instrument: str, is_intraday: bool, start_time: str,
                 end_time: str,
                 stop_loss: Union[int, float] = None, target: Union[int, float] = None,
                 per_trade_commission: float = 0, slippage: float = 0,
                 re_entry_count: int = 0, re_execute_count: int = 0,
                 timeframe: str = "1min", move_sl_to_cost: bool = False,
                 trading_days_before_expiry=None, how_far_otm_hedge_point: int = None,
                 how_far_otm_short_point: int = None, expiry_week: int = 0):
        self.start_date = start_date
        self.end_date = end_date
        self.instrument = instrument
        self.is_intraday = is_intraday
        self.start_time = utils.str_to_time_obj(start_time)
        self.end_time = utils.str_to_time_obj(end_time)
        self.stop_loss = stop_loss
        self.target = target
        self.per_trade_commission = per_trade_commission
        self.slippage = slippage
        self.re_entry_count = re_entry_count
        self.re_execute_count = re_execute_count
        self.timeframe = timeframe
        self.move_sl_to_cost = move_sl_to_cost
        self.trading_days_before_expiry = trading_days_before_expiry
        self.how_far_otm_hedge_point = how_far_otm_hedge_point
        self.how_far_otm_short_point = how_far_otm_short_point
        self.expiry_week = expiry_week
