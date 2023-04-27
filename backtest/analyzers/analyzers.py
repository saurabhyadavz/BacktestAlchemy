import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import calendar
import os

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


class DegenPlotter:

    def __init__(self, day_points: pd.Series, lot_size: int, strat_name: str = ""):
        self.curve_plot_path = os.path.join(os.getcwd(), f"{strat_name}_curve.png")
        self.monthly_report_plot_path = os.path.join(os.getcwd(), f"{strat_name}_monthly_report.png")
        self.day_pnl_df = pd.DataFrame(list(day_points.items()), columns=['date', 'points'])
        self.day_pnl_df['points'] = self.day_pnl_df['points'] * lot_size
        self.calculate_cumulative_sum()
        self.calculate_drawdown()

    def calculate_cumulative_sum(self):
        self.day_pnl_df['cumulative'] = self.day_pnl_df['points'].cumsum()

    def calculate_drawdown(self):
        self.calculate_cumulative_sum()
        self.day_pnl_df['peak'] = self.day_pnl_df['cumulative'].cummax()
        self.day_pnl_df['max_drawdown'] = (self.day_pnl_df['cumulative'] - self.day_pnl_df['peak'])

    def plot_curve(self):
        fig, ax = plt.subplots(nrows=2, sharex=True, figsize=(20, 15))
        sns.lineplot(x='date', y='cumulative', data=self.day_pnl_df, ax=ax[0])
        ax[0].set(title='Cumulative Sum and Drawdown Plot', ylabel='Cumulative Sum')
        sns.lineplot(x='date', y='max_drawdown', data=self.day_pnl_df, color='r', ax=ax[1])
        ax[1].set(ylabel='Drawdown')
        plt.xlabel('Date')
        plt.subplots_adjust(hspace=0.01)
        plt.savefig(self.curve_plot_path)

    def plot_monthly_report(self):
        self.day_pnl_df['date'] = pd.to_datetime(self.day_pnl_df['date'])
        self.day_pnl_df['month'] = self.day_pnl_df['date'].dt.month
        self.day_pnl_df['year'] = self.day_pnl_df['date'].dt.year
        monthly_returns = self.day_pnl_df.groupby(['year', 'month'])['points'].sum().reset_index()
        monthly_returns['month_name'] = monthly_returns['month'].apply(lambda x: calendar.month_name[x])
        month_order = [calendar.month_name[i] for i in range(1, 13)]
        monthly_returns['month_name'] = pd.Categorical(monthly_returns['month_name'], categories=month_order,
                                                       ordered=True)
        monthly_returns.sort_values(['year', 'month_name'], inplace=True)
        monthly_returns_pivot = monthly_returns.pivot(index='year', columns='month_name', values='points')
        plt.figure(figsize=(20, 15))
        sns.heatmap(data=monthly_returns_pivot, cmap='RdYlGn', annot=True, fmt='.2f')
        plt.title('Total Returns Month-wise')
        plt.xlabel('Month')
        plt.ylabel('Year')
        plt.savefig(self.monthly_report_plot_path)

    def plot_all(self):
        self.plot_curve()
        self.plot_monthly_report()
