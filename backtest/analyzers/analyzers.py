import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import calendar
import os
import matplotlib.ticker as mtick
from backtest.strategy.strategy import Strategy
from backtest.utils.utils import get_instrument_lot_size, calculate_leverage, NOTIONAL_VALUE_ASSUMED


def get_x_freq(df: pd.DataFrame) -> str:
    """Returns frequency of date based on df"""
    n = len(df)
    if n > 800:
        freq = "Y"
    elif n > 150:
        freq = 'M'

    elif 100 <= n < 150:
        freq = '20D'
    elif 50 <= n < 100:
        freq = '10D'
    elif 25 <= n < 50:
        freq = '5D'
    else:
        freq = 'D'
    return freq


class Analyzers:
    def __init__(self, capital: int, lots: int, instrument: str, start_date: str, end_date: str,
                 strat_name: str, slippage: float = 0):
        self.capital = capital
        self.lots = lots
        self.instrument = instrument
        self.strat_name = strat_name
        self.start_date = start_date
        self.end_date = end_date
        self.position_size = get_instrument_lot_size(instrument) * lots
        self.leverage = calculate_leverage(capital, lots)
        self.slippage = slippage
        self.strat_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), strat_name)
        self.metrics_path = os.path.join(self.strat_dir, "metrics.csv")
        self.tradebook_path = os.path.join(self.strat_dir, f"{self.strat_name}_tradebook.csv")
        self.tradebook_df = pd.read_csv(self.tradebook_path)
        self.daily_pnl = None
        self.daily_pct = None
        self.calculate_metrices()
        self.plot_pnl_curve()
        self.plot_monthly_heatmap(figsize=(8, 5))

    def plot_monthly_heatmap(self, figsize=None):
        df = self.daily_pct
        df["month"] = df["date"].dt.month
        df["year"] = df["date"].dt.year
        monthly_returns = df.groupby(['year', 'month'])['pnl_pct'].sum().reset_index()
        monthly_returns['month_name'] = monthly_returns['month'].apply(lambda x: calendar.month_abbr[x])
        month_order = [calendar.month_abbr[i] for i in range(1, 13)]
        print(month_order)
        monthly_returns['month_name'] = pd.Categorical(monthly_returns['month_name'], categories=month_order,
                                                       ordered=True)
        monthly_returns.sort_values(['year', 'month_name'], inplace=True)
        monthly_returns_pivot = monthly_returns.pivot(index='year', columns='month_name', values='pnl_pct')
        monthly_returns_pivot.columns = map(lambda x: str(x).upper(), monthly_returns_pivot.columns)
        monthly_returns_pivot.columns.name = None
        fig_height = len(monthly_returns_pivot) / 3
        if figsize is None:
            size = list(plt.gcf().get_size_inches())
            figsize = (size[0], size[1])
        figsize = (figsize[0], max([fig_height, figsize[1]]))
        fig, ax = plt.subplots(figsize=figsize)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)

        fig.set_facecolor('white')
        ax.set_facecolor('white')
        ax.set_title('      Monthly Returns (%)\n', fontsize=14, y=.995,
                     fontname="Arial", fontweight='bold', color='black')
        ax = sns.heatmap(monthly_returns_pivot, ax=ax, annot=True, center=0,
                         annot_kws={"size": 10},
                         fmt="0.2f", linewidths=0.5,
                         square=True, cbar=False, cmap="RdYlGn",
                         cbar_kws={'format': '%.0f%%'})
        ax.set_ylabel('Years', fontname="Arial",
                      fontweight='bold', fontsize=12)
        ax.yaxis.set_label_coords(-.1, .5)
        ax.tick_params(colors="#808080")
        plt.xticks(rotation=0, fontsize=10 * 1.2)
        plt.yticks(rotation=0, fontsize=10 * 1.2)
        try:
            plt.subplots_adjust(hspace=0, bottom=0, top=1)
        except Exception:
            pass
        try:
            fig.tight_layout(w_pad=0, h_pad=0)
        except Exception:
            pass
        plt.savefig(os.path.join(self.strat_dir, "monthly_returns.png"))

    def plot_pnl_curve(self):
        df = self.daily_pnl.copy()
        # ROC
        df["pnl_pct"] = (df["pnl"] / self.capital) * 100
        df.to_csv("check.csv")
        self.daily_pct = df.copy()
        # ROC Cummulative
        df["pnl_pct_cumulative"] = df["pnl_pct"].cumsum()
        df.at[0, "pnl"] = self.capital + df.at[0, "pnl"]
        cumulative_pnl = np.cumsum(df['pnl'])
        max_cumulative_pnl = np.maximum.accumulate(cumulative_pnl)
        df["drawdown"] = ((cumulative_pnl / max_cumulative_pnl) - 1) * 100
        df.set_index("date", inplace=True)
        plt.figure(figsize=(15, 10))
        plt.fill_between(df.index, df["pnl_pct_cumulative"], color='green', alpha=0.5)
        plt.fill_between(df.index, df["drawdown"], color='red', alpha=0.5)

        fmt = '%.0f%%'
        yticks = mtick.FormatStrFormatter(fmt)
        plt.gca().yaxis.set_major_formatter(yticks)

        max_drawdown = df['drawdown'].min()
        max_pnl_pct_cumulative = df['pnl_pct_cumulative'].max()

        # Set y-axis ticks
        yticks_values = [max_drawdown, max_pnl_pct_cumulative, 0]
        yticks_labels = [f"{round(max_drawdown, 2)}%", f"{round(max_pnl_pct_cumulative, 2)}%", "0"]

        # add ticks on Y at interval of max_pnl_pct_cumulative/6
        max_pnl = int(max_pnl_pct_cumulative)
        start_pnl_intrvl = max_pnl // 6
        if start_pnl_intrvl != 0:
            pct_cumulative_ticks = [i for i in range(start_pnl_intrvl, max_pnl + 1, start_pnl_intrvl)]
            pct_cumulative_ticklabels = [f"{round(x, 2)}%" for x in pct_cumulative_ticks]
            yticks_values.extend(pct_cumulative_ticks)
            yticks_labels.extend(pct_cumulative_ticklabels)

        # add ticks on Y at interval of max_drawdown/2
        max_dd = abs(int(max_drawdown))
        start_dd_intrvl = max_dd // 2
        if start_dd_intrvl != 0:
            max_dd_ticks = [-1 * i for i in range(start_dd_intrvl, max_dd, start_dd_intrvl)]
            mad_dd_ticklabels = [f"-{round(x, 2)}%" for x in max_dd_ticks]
            yticks_values.extend(max_dd_ticks)
            yticks_labels.extend(mad_dd_ticklabels)
        x_freq = get_x_freq(df)
        date_ticks = pd.date_range(df.index.min(), df.index.max(), freq=x_freq)
        date_labels = date_ticks.strftime('%d %b-%y')
        if x_freq == "M":
            date_labels = date_ticks.strftime('%b-%y')
        if x_freq == "Y":
            date_labels = date_ticks.strftime('%y')

        plt.yticks(yticks_values, yticks_labels)
        plt.xticks(date_ticks, date_labels)

        # Draw horizontal lines at the y-coordinates for Max Drawdown and Max Drawdown
        plt.axhline(max_drawdown, color='red', linestyle='--', label='Max Drawdown')
        plt.axhline(0, color='black', linestyle='--', alpha=0.5)
        plt.axhline(max_pnl_pct_cumulative, color='green', linestyle='--', label='Max PnL(%)')

        plt.xticks(rotation=25, fontsize=15)
        plt.yticks(fontsize=15)

        # plt.legend(loc="center left")
        plt.legend(loc='upper left', bbox_to_anchor=(0, 0.9), fontsize=15)
        plt.xlabel("Date", fontsize=15)
        plt.ylabel("P&L", fontsize=15)
        plt.title("P&L vs Drawdown", fontsize=15)
        plt.grid(True, alpha=0.2)
        plt.savefig(os.path.join(self.strat_dir, "pnl_dd_plot.png"))

    def max_drawdown(self, daily_pnl: pd.DataFrame):
        """Calculates the maximum drawdown"""
        cumulative_points = np.cumsum(daily_pnl['pnl'])
        max_cumulative_points = np.maximum.accumulate(cumulative_points)
        drawdown = max_cumulative_points - cumulative_points
        max_drawdown = np.max(drawdown)
        max_drawdown_pct = np.max(drawdown / max_cumulative_points) * 100
        return max_drawdown, max_drawdown_pct

    def calculate_metrices(self):
        """Calculates Metrices from tradebook"""

        tradebook_df = self.tradebook_df.copy()
        tradebook_df["datetime"] = pd.to_datetime(tradebook_df["datetime"])
        tradebook_df["date"] = tradebook_df["datetime"].dt.date
        tradebook_df["price_with_slippage"] = np.where(
            tradebook_df["side"] == 1, tradebook_df["price"] * tradebook_df["side"] * (1 + self.slippage),
            tradebook_df["price"] * tradebook_df["side"] * (1 - self.slippage)
        )
        unable_to_trade_days = tradebook_df.iloc[0]["unable_to_trade_days"]
        tradebook_df.drop(["unable_to_trade_days", "is_intraday"], inplace=True, axis=1)
        tradebook_grp = tradebook_df.groupby("date")
        day_points_df = pd.DataFrame(columns=["date", "points", "total trade"])
        for date, grp in tradebook_grp:
            points = grp["price_with_slippage"].sum() * -1
            total_trades = grp.shape[0]
            day_points_df = pd.concat([day_points_df, pd.DataFrame({"date": [date], "points": [points],
                                                                    "total trade": total_trades})], ignore_index=True)
        day_points_df["date"] = pd.to_datetime(day_points_df["date"])
        day_points_df["pnl"] = day_points_df["points"] * self.position_size
        self.daily_pnl = day_points_df.copy()
        day_points_df.at[0, "pnl"] = self.capital + day_points_df.at[0, "pnl"]
        day_points_df.set_index("date", inplace=True)
        day_points_df = day_points_df.dropna()

        # Calculate total trades
        total_trades = len(day_points_df)

        # Calculate total wins
        total_wins = len(day_points_df[day_points_df['points'] > 0])

        # Calculate total losses
        total_losses = len(day_points_df[day_points_df['points'] < 0])

        # Calculate total points
        total_points = np.sum(day_points_df['points'])

        # Calculate max win
        max_win = np.max(day_points_df['points'])

        # Calculate max loss
        max_loss = np.min(day_points_df['points'])

        max_dd, max_dd_pct = self.max_drawdown(day_points_df)

        # Calculate calmar
        calmar = (total_points * self.position_size / max_dd) if max_dd > 0 else 0

        # Calculate win rate
        win_rate = (total_wins / total_trades)

        # Calculate RR, avg win points, avg loss points, profit factor
        win_pts = np.sum(day_points_df[day_points_df['points'] > 0]['points'])
        loss_pts = np.sum(day_points_df[day_points_df['points'] < 0]['points'])
        OA_adj_pts = (win_pts - max_win)
        avg_points_on_winning_days = np.mean(day_points_df[day_points_df['points'] > 0]['points'])
        avg_loss_on_losing_days = np.mean(day_points_df[day_points_df['points'] < 0]['points'])
        avg_points_winning_days_oa_adj = round(OA_adj_pts / total_wins, 1)
        risk_to_reward = round(abs(avg_points_on_winning_days / avg_loss_on_losing_days), 2)
        profit_factor = round(abs(win_pts / loss_pts), 2)
        outlier_adjusted_profit_factor = round(abs(OA_adj_pts / loss_pts), 2)
        expectancy = (win_rate * risk_to_reward) - (1 - win_rate)

        points_mean = np.mean(day_points_df['points'])

        # Calculate max loss day
        max_loss_day = day_points_df[day_points_df['points'] == max_loss].index[0]

        # Calculate max win day
        max_win_day = day_points_df[day_points_df['points'] == max_win].index[0]

        # Calculate sharpe ratio
        sharpe = (np.sqrt(252) * points_mean) / np.std(day_points_df['points'])

        # Calculate sortino ratio
        down_dev = np.where(day_points_df['points'] < 0, day_points_df['points'], 0).std()
        sortino = (np.sqrt(252) * points_mean) / down_dev

        # Calculate average monthly ROI
        monthly_returns = day_points_df['points'].resample('M').sum()
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
            'Max Drawdown (Rs)': round(max_dd, 2),
            'Max Drawdown (%)': round(max_dd_pct, 2),
            'Max Drawdown Days': f"{None}[{None} to {None}]",
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
        metrics.to_csv(self.metrics_path, index_label='Test Start Date')
