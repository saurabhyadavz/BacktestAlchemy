import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import calendar
import os
from backtest.utils.utils import get_position_size_and_margin


class Analyzers:
    def __init__(self, capital: int, instrument: str):
        self.capital = capital
        self.position_size, self.margin_required = get_position_size_and_margin(capital, instrument)

    def get_new_matrices(self, daily_points_series: pd.Series, strat='', unable_to_trade_days: int = 0):
        df = pd.DataFrame(daily_points_series, columns=['points'])
        df = df.reset_index()
        df.columns = ['date', 'points']
        df["date"] = pd.to_datetime(df["date"])
        df.set_index("date", inplace=True)
        df = df.dropna()

        # Calculate total trades
        total_trades = len(df)

        # Calculate total wins
        total_wins = len(df[df['points'] > 0])

        # Calculate total losses
        total_losses = len(df[df['points'] < 0])

        # Calculate total points
        total_points = np.sum(df['points'])

        # Calculate max win
        max_win = np.max(df['points'])

        # Calculate max loss
        max_loss = np.min(df['points'])

        # Calculate max drawdown
        cumulative_points = np.cumsum(df['points'])
        max_cumulative_points = np.maximum.accumulate(cumulative_points)
        max_drawdown = np.max(max_cumulative_points - cumulative_points)

        # Calculate max drawdown percentage

        max_drawdown_percentage = np.max((max_cumulative_points - cumulative_points) / max_cumulative_points) * 100

        # Calculate calmar
        calmar = (total_points / max_drawdown) if max_drawdown > 0 else 0

        # Calculate win rate
        win_rate = (total_wins / total_trades) * 100

        # Calculate average points on winning days
        avg_points_on_winning_days = np.mean(df[df['points'] > 0]['points'])

        # Calculate average loss on losing days
        avg_loss_on_losing_days = np.mean(df[df['points'] < 0]['points'])

        points_mean = np.mean(df['points'])

        # Calculate max loss day
        max_loss_day = df[df['points'] == max_loss].index[0]

        # Calculate max win day
        max_win_day = df[df['points'] == max_win].index[0]

        # Calculate sharpe ratio
        sharpe = (np.sqrt(252) * points_mean) / np.std(df['points'])

        # Calculate sortino ratio
        down_dev = np.where(df['points'] < 0, df['points'], 0).std()
        sortino = (np.sqrt(252) * points_mean) / down_dev

        # Calculate average monthly ROI
        monthly_returns = df['points'].resample('M').sum()
        average_monthly_roi = np.mean(monthly_returns / total_points) * 100
        metrics = pd.DataFrame(columns=['Strategy', 'Total Capital', 'Margin Used', 'Total Trading days',
                                        'Win days', 'Loss Days', 'Win Rate (%)', 'Average Monthly ROI (%)',
                                        'Total Profit (Rs)', 'Max Profit (Rs)', 'Max Loss (Rs)', 'Max Winning Day',
                                        'Max Losing Day', 'Max Drawdown (Rs)', 'Max Drawdown (%)', 'Calmar',
                                        'Sharpe Ratio', 'Sortino Ratio', 'Unable to trade days']
                               )
        metrics = pd.concat([metrics, pd.DataFrame({'Strategy': strat,
                                                    'Total Capital': self.capital,
                                                    'Margin Used': self.margin_required,
                                                    'Total Trading days': total_trades,
                                                    'Win days': total_wins,
                                                    'Loss Days': total_losses,
                                                    'Win Rate (%)': round(win_rate, 2),
                                                    'Average Monthly ROI (%)': average_monthly_roi,
                                                    'Total Profit (Rs)': round(total_points * self.position_size, 2),
                                                    'Max Profit (Rs)': round(max_win * self.position_size, 2),
                                                    'Max Loss (Rs)': round(max_loss * self.position_size, 2),
                                                    'Max Winning Day': max_win_day.date(),
                                                    'Max Losing Day': max_loss_day.date(),
                                                    'Max Drawdown (Rs)': round(max_drawdown * self.position_size, 2),
                                                    'Max Drawdown (%)': round(max_drawdown_percentage, 2),
                                                    'Calmar': round(calmar, 2),
                                                    'Sharpe Ratio': round(sharpe, 2),
                                                    'Sortino Ratio': round(sortino, 2),
                                                    'Unable to trade days': unable_to_trade_days}, index=[0])],
                            ignore_index=True)
        return metrics.T

    def get_metrics(self, daily_points_series: pd.Series, strat='', unable_to_trade_days: int = 0):
        df = pd.DataFrame(daily_points_series, columns=['points'])
        df = df.reset_index()
        df.columns = ['Date', 'points']
        print(df)
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
        win_pts = np.where(ret > 0, ret, 0).sum()
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
                                        'Sharpe', 'Sortino', 'Max Win', 'Max Loss', 'Avg Win', 'Avg Loss',
                                        'Unable to trade days'])
        metrics = pd.concat([metrics, pd.DataFrame({'Strategy': col,
                                                    'Total Trades': tot_trades,
                                                    'Total Points': tot_pts * self.lot_size,
                                                    'Wins': tot_wins,
                                                    'Losses': tot_losses,
                                                    'Win Rate': win_rate,
                                                    'Pts per Trade': avg_pts * self.lot_size,
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
                                                    'Avg Loss': avg_loss,
                                                    'Unable to trade days': unable_to_trade_days}, index=[0])],
                            ignore_index=True)
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
        cmap = sns.color_palette(["red", "lightgreen"])
        sns.heatmap(data=monthly_returns_pivot, cmap=cmap, annot=True, fmt='.2f', center=0,
                    vmin=monthly_returns_pivot.values.min(), vmax=monthly_returns_pivot.values.max(),
                    linecolor='white', linewidths=1, square=True, xticklabels=True, cbar=False)
        plt.title('Total Returns Month-wise')
        plt.xlabel('Month')
        plt.ylabel('Year')
        plt.savefig(self.monthly_report_plot_path)

    def plot_all(self):
        self.plot_curve()
        self.plot_monthly_report()
