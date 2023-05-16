import os
from typing import Union
from backtest.utils import utils
from backtest.utils.utils import get_instrument_lot_size


class Strategy:
    def __init__(self, strat_name: str, start_date: str, end_date: str, instrument: str, is_intraday: bool,
                 start_time: str, end_time: str, capital: int, lots: int, stoploss_mtm_points: Union[int, float] = None,
                 target: Union[int, float] = None, per_trade_commission: float = 0,
                 re_entry_count: int = 0, re_execute_count: int = 0, stoploss_mtm_rupees: int = None,
                 timeframe: str = "1min", opt_timeframe: str = "1min", move_sl_to_cost: bool = False,
                 trading_days_before_expiry=None, how_far_otm_hedge_point: int = None,
                 how_far_otm_short_point: int = None, expiry_week: int = 0, stoploss_pct: float = None,
                 buffer: float = 0, close_half_on_mtm_rupees: int = None):
        self.strat_name = strat_name
        self.start_date = start_date
        self.end_date = end_date
        self.instrument = instrument
        self.is_intraday = is_intraday
        self.start_time = utils.str_to_time_obj(start_time)
        self.end_time = utils.str_to_time_obj(end_time)
        self.stoploss_pct = stoploss_pct
        self.stoploss_mtm_points = -1 * stoploss_mtm_points if stoploss_mtm_points is not None else stoploss_mtm_points
        self.stoploss_mtm_rupees = -1 * stoploss_mtm_rupees if stoploss_mtm_rupees is not None else stoploss_mtm_rupees
        self.target = target
        self.per_trade_commission = per_trade_commission
        self.re_entry_count = re_entry_count
        self.re_execute_count = re_execute_count
        self.timeframe = timeframe
        self.opt_timeframe = opt_timeframe
        self.move_sl_to_cost = move_sl_to_cost
        self.trading_days_before_expiry = trading_days_before_expiry
        self.how_far_otm_hedge_point = how_far_otm_hedge_point
        self.how_far_otm_short_point = how_far_otm_short_point
        self.expiry_week = expiry_week
        self.capital = capital
        self.lots = lots
        self.position_size = get_instrument_lot_size(instrument) * lots
        self.buffer = buffer
        self.close_half_on_mtm_rupees = close_half_on_mtm_rupees
        self.create_strat_dir()

    def create_strat_dir(self):
        curr_dir = os.path.dirname(os.path.dirname(__file__))
        strategy_dir = os.path.join(curr_dir, self.strat_name)
        if not os.path.exists(strategy_dir):
            os.mkdir(strategy_dir)