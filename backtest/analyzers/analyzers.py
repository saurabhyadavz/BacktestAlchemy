import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import calendar
import os
from backtest.utils.utils import get_instrument_lot_size, calculate_leverage, NOTIONAL_VALUE_ASSUMED


class Analyzers:
    def __init__(self, capital: int, lots: int, instrument: str, start_date: str, end_date: str, strat_name: str = ''):
        self.capital = capital
        self.lots = lots
        self.instrument = instrument
        self.strat_name = strat_name
        self.start_date = start_date
        self.end_date = end_date
        self.position_size = get_instrument_lot_size(instrument) * lots
        self.leverage = calculate_leverage(capital, lots)

    def get_matrices(self, daily_points_series: pd.Series, unable_to_trade_days: int = 0):
        df = pd.DataFrame(daily_points_series, columns=['points'])
        df = df.reset_index()
        df.columns = ['date', 'points']
        df["date"] = pd.to_datetime(df["date"])
        df["points"] = df["points"]
        df["points_1"] = df["points"] * self.position_size
        df.at[0, "points_1"] = self.capital + df.at[0, "points_1"]
        df.to_csv("check.csv")
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
        cumulative_points = np.cumsum(df['points_1'])
        max_cumulative_points = np.maximum.accumulate(cumulative_points)
        drawdown = max_cumulative_points - cumulative_points
        max_drawdown = np.max(drawdown)
        max_drawdown_days = len(drawdown[drawdown == max_drawdown])
        max_drawdown_days_idx = np.where(drawdown == max_drawdown)[0]
        max_drawdown_start_day = (df.iloc[max_drawdown_days_idx[0]].name).date()
        max_drawdown_end_day = (df.iloc[max_drawdown_days_idx[-1]].name).date()

        # Calculate max drawdown percentage

        max_drawdown_percentage = np.max(drawdown / max_cumulative_points) * 100

        # Calculate calmar
        calmar = (total_points * self.position_size / max_drawdown) if max_drawdown > 0 else 0

        # Calculate win rate
        win_rate = (total_wins / total_trades)

        # Calculate RR, avg win points, avg loss points, profit factor
        win_pts = np.sum(df[df['points'] > 0]['points'])
        loss_pts = np.sum(df[df['points'] < 0]['points'])
        OA_adj_pts = (win_pts - max_win)
        avg_points_on_winning_days = np.mean(df[df['points'] > 0]['points'])
        avg_loss_on_losing_days = np.mean(df[df['points'] < 0]['points'])
        avg_points_winning_days_oa_adj = round(OA_adj_pts / total_wins, 1)
        risk_to_reward = round(abs(avg_points_on_winning_days / avg_loss_on_losing_days), 2)
        profit_factor = round(abs(win_pts / loss_pts), 2)
        outlier_adjusted_profit_factor = round(abs(OA_adj_pts / loss_pts), 2)
        expectancy = (win_rate * risk_to_reward) - (1 - win_rate)

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
        metrics = pd.DataFrame(columns=['Test Start Date', 'Test End Date', 'Instrument',
                                        'Strategy', 'Total Capital', 'Notional Value Asm.', 'Leverage',
                                        'Traded with Lots', 'Total Trading days', 'Win days', 'Loss Days',
                                        'Win Rate (%)', 'Avg Profit on Profit Days Outlier Adjusted (Rs)',
                                        'Avg Profit on Profit Days (Rs)', 'Avg Loss on Loss Days (Rs)',
                                        'Average Monthly ROI (%)', 'Total Profit (Rs)', 'Max Profit (Rs)',
                                        'Max Loss (Rs)', 'Max Winning Day', 'Max Losing Day', 'Max Drawdown (Rs)',
                                        'Max Drawdown (%)', 'Max Drawdown Days', 'Risk to reward', 'Profit Factor',
                                        'Outlier Adjusted Profit Factor', 'Expectancy',
                                        'Calmar', 'Sharpe Ratio (Annualised)', 'Sortino Ratio (Annualised)',
                                        'Unable to trade days'])
        metrics = pd.concat([metrics, pd.DataFrame({
            'Test Start Date': self.start_date,
            'Test End Date': self.end_date,
            'Instrument': self.instrument,
            'Strategy': self.strat_name,
            'Total Capital': self.capital,
            'Notional Value Asm.': NOTIONAL_VALUE_ASSUMED,
            'Leverage': f"{self.leverage}x",
            'Traded with Lots': self.lots,
            'Total Trading days': total_trades,
            'Win days': total_wins,
            'Loss Days': total_losses,
            'Win Rate (%)': round(win_rate * 100, 2),
            'Avg Profit on Profit Days Outlier Adjusted (Rs)': round(
                avg_points_winning_days_oa_adj * self.position_size, 2),
            'Avg Profit on Profit Days (Rs)': round(
                avg_points_on_winning_days * self.position_size, 2),
            'Avg Loss on Loss Days (Rs)': round(
                avg_loss_on_losing_days * self.position_size, 2),
            'Average Monthly ROI (%)': round(average_monthly_roi, 2),
            'Total Profit (Rs)': round(total_points * self.position_size, 2),
            'Max Profit (Rs)': round(max_win * self.position_size, 2),
            'Max Loss (Rs)': round(max_loss * self.position_size, 2),
            'Max Winning Day': max_win_day.date(),
            'Max Losing Day': max_loss_day.date(),
            'Max Drawdown (Rs)': round(max_drawdown * self.position_size, 2),
            'Max Drawdown (%)': round(max_drawdown_percentage, 2),
            'Max Drawdown Days': f"{max_drawdown_days}[{max_drawdown_start_day} to {max_drawdown_end_day}]",
            'Risk to reward': risk_to_reward,
            'Profit Factor': profit_factor,
            'Outlier Adjusted Profit Factor': outlier_adjusted_profit_factor,
            'Expectancy': round(expectancy, 2),
            'Calmar': round(calmar, 2),
            'Sharpe Ratio (Annualised)': round(sharpe, 2),
            'Sortino Ratio (Annualised)': round(sortino, 2),
            'Unable to trade days': unable_to_trade_days}, index=[0])], ignore_index=True)
        metrics.reset_index(drop=True, inplace=True)
        metrics = metrics.set_index('Test Start Date').T
        return metrics


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
