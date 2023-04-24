import pandas as pd
import numpy as np


class Analyzers:
    def get_metrics(self, df, strat=''):
        df = df.dropna()
        ret = df.values
        col = df.name

        if strat == '':
            col = col
        else:
            col = strat

        tot_trades = len(ret)
        tot_wins = np.sum(ret > 0)
        tot_losses = np.sum(ret < 0)

        tot_pts = ret.sum()
        win_pts = np.where(ret > 0, ret, 0).sum()
        loss_pts = np.where(ret < 0, ret, 0).sum()

        max_win = ret.max()
        max_loss = ret.min()

        sigma = ret.std()
        mu = ret.mean()
        down_dev = np.where(ret < 0, ret, 0).std()

        cum_pts = ret.cumsum()
        cum_max = np.maximum.accumulate(cum_pts)
        dd = cum_pts - cum_max
        max_dd = dd.min()
        calmar = round(-tot_pts / max_dd, 2)
        #     dd_pct = dd/df2['Entry_Price']
        OA_adj_pts = (win_pts - max_win)

        win_rate = round(tot_wins / tot_trades, 3)
        avg_win = round(OA_adj_pts / tot_wins, 1)
        avg_loss = round(loss_pts / tot_losses, 1)
        avg_pts = (tot_pts - max_win) / tot_trades
        RR = round(abs(avg_win / avg_loss), 2)
        PF = round(abs(win_pts / loss_pts), 2)
        OAPF = round(abs(OA_adj_pts / loss_pts), 2)

        exp = round(win_rate * RR - (1 - win_rate), 2)
        exp_pts = win_rate * avg_win + (1 - win_rate) * avg_loss
        kelly = round(win_rate - ((1 - win_rate) / RR), 3)

        sharpe = round(np.sqrt(252) * mu / sigma, 2)
        sortino = round(np.sqrt(252) * mu / down_dev, 2)

        metrics = pd.DataFrame(columns=['Strategy', 'Total Trades', 'Total Points', 'Wins', 'Losses', 'Win Rate', 'RR',
                                        'PF', 'OAPF', 'Pts per Trade', 'Exp_in_R', 'Kelly', 'Max DD', 'CALMAR',
                                        'Sharpe', 'Sortino', 'Max Win', 'Max Loss', 'Avg Win', 'Avg Loss'])
        metrics = pd.concat([metrics, pd.DataFrame({'Strategy': col,
                                                    'Total Trades': tot_trades,
                                                    'Total Points': tot_pts,
                                                    'Wins': tot_wins,
                                                    'Losses': tot_losses,
                                                    'Win Rate': win_rate,
                                                    'Pts per Trade': avg_pts,
                                                    'RR': RR,
                                                    'PF': PF,
                                                    'OAPF': OAPF,
                                                    'Exp_in_R': exp,
                                                    'Kelly': kelly,
                                                    'CALMAR': calmar,
                                                    'Max DD': max_dd,
                                                    'Sharpe': sharpe,
                                                    'Sortino': sortino,
                                                    'Max Win': max_win,
                                                    'Max Loss': max_loss,
                                                    'Avg Win': avg_win,
                                                    'Avg Loss': avg_loss}, index=[0])], ignore_index=True)
        return metrics.T
