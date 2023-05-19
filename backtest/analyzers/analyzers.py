import pandas as pd
import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import calendar
import os
import matplotlib.ticker as mtick
from fpdf import FPDF
import csv

from backtest.utils.utils import get_instrument_lot_size, calculate_leverage, NOTIONAL_VALUE_ASSUMED


def get_x_freq(df: pd.DataFrame) -> str:
    """Returns frequency of date based on df"""
    n = len(df)
    if n > 800:
        freq = "Y"
    elif n > 250:
        freq = '3M'
    elif n > 150:
        freq = 'M'
    elif 100 <= n < 150:
        freq = '20D'
    elif 50 <= n < 100:
        freq = '10D'
    elif 25 <= n < 50:
        freq = '5D'
    elif 10 <= n < 25:
        freq = '2D'
    else:
        freq = 'D'
    return freq


def remove_outliers(returns, quantile=.95):
    """Returns series of returns without the outliers"""
    return returns[returns < returns.quantile(quantile)]


class Analyzers:
    def __init__(self, capital: int, lots: int, instrument: str, start_date: str, end_date: str,
                 strat_name: str, slippage: float = 0):
        self.capital = capital
        self.lots = lots
        self.instrument = instrument
        self.strat_name = strat_name
        self.start_date = start_date
        self.end_date = end_date
        self.leverage = calculate_leverage(capital, lots)
        self.slippage = slippage
        self.strat_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), strat_name)
        self.metrics_path = os.path.join(self.strat_dir, "metrics.csv")
        self.daily_summary_path = os.path.join(self.strat_dir, "daily_summary.csv")
        self.tradebook_path = os.path.join(self.strat_dir, f"{self.strat_name}_tradebook.csv")
        self.tradebook_df = pd.read_csv(self.tradebook_path)
        self.unable_to_trade_days = 0
        self.daily_pnl = self.get_daily_pnl()
        self.daily_returns = self.get_daily_returns()
        self.calculate_metrices()
        self.drawdown_details()
        self.plot_pnl_curve()
        self.plot_drawdown_curve()
        self.plot_monthly_heatmap(figsize=(8, 5))
        self.calculate_daily_summary()
        self.prepare_pdf_report()

    def get_daily_returns(self):
        daily_returns_df = self.daily_pnl.copy()
        daily_returns_df["pnl_pct"] = (daily_returns_df["pnl"] / self.capital) * 100
        return daily_returns_df

    def get_daily_pnl(self):
        tradebook_df = pd.read_csv(self.tradebook_path)
        tradebook_df["datetime"] = pd.to_datetime(tradebook_df["datetime"])
        tradebook_df["date"] = tradebook_df["datetime"].dt.date
        tradebook_df["price_with_slippage"] = np.where(
            tradebook_df["side"] == 1,
            tradebook_df["price"] * tradebook_df["side"] * (1 + self.slippage),
            tradebook_df["price"] * tradebook_df["side"] * (1 - self.slippage)
        )
        self.unable_to_trade_days = tradebook_df.iloc[0]["unable_to_trade_days"]
        tradebook_df.drop(["unable_to_trade_days", "is_intraday"], inplace=True, axis=1)
        tradebook_grp = tradebook_df.groupby("date")
        day_points_df = pd.DataFrame(columns=["date", "points", "pnl", "total trade"])
        for date, grp in tradebook_grp:
            pnl = (grp["price_with_slippage"] * grp["traded_quantity"]).sum() * -1
            points = grp["price_with_slippage"].sum() * -1
            total_trades = grp.shape[0]
            day_points_df = pd.concat([day_points_df, pd.DataFrame({"date": [date], "points": [points], "pnl": [pnl],
                                                                    "total trade": total_trades})], ignore_index=True)
        day_points_df["date"] = pd.to_datetime(day_points_df["date"])
        day_points_df = day_points_df.dropna()
        return day_points_df

    def plot_monthly_heatmap(self, figsize=None):
        df = self.daily_returns.copy()
        df["month"] = df["date"].dt.month
        df["year"] = df["date"].dt.year
        monthly_returns = df.groupby(['year', 'month'])['pnl_pct'].sum().reset_index()
        monthly_returns['month_name'] = monthly_returns['month'].apply(lambda x: calendar.month_abbr[x])
        month_order = [calendar.month_abbr[i] for i in range(1, 13)]
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
        plt.close(fig)

    def plot_drawdown_curve(self):
        daily_returns_df = self.daily_returns.copy()
        daily_returns_df["pnl_pct_cumulative"] = daily_returns_df["pnl_pct"].cumsum()
        daily_returns_df.at[0, "pnl"] = self.capital + daily_returns_df.at[0, "pnl"]
        cumulative_pnl = np.cumsum(daily_returns_df['pnl'])
        max_cumulative_pnl = np.maximum.accumulate(cumulative_pnl)
        daily_returns_df["drawdown"] = ((cumulative_pnl / max_cumulative_pnl) - 1) * 100
        daily_returns_df.set_index("date", inplace=True)
        plt.figure(figsize=(15, 10))
        plt.fill_between(daily_returns_df.index, daily_returns_df["drawdown"], color='red', alpha=0.5)
        fmt = '%.0f%%'
        yticks = mtick.FormatStrFormatter(fmt)
        plt.gca().yaxis.set_major_formatter(yticks)

        max_drawdown = daily_returns_df['drawdown'].min()

        yticks_values = [max_drawdown, 0]
        yticks_labels = [f"{round(max_drawdown, 2)}%", "0"]

        max_dd = abs(int(max_drawdown))
        start_dd_intrvl = max_dd // 2
        if start_dd_intrvl != 0:
            max_dd_ticks = [-1 * i for i in range(start_dd_intrvl, max_dd, start_dd_intrvl)]
            mad_dd_ticklabels = [f"-{round(x, 2)}%" for x in max_dd_ticks]
            yticks_values.extend(max_dd_ticks)
            yticks_labels.extend(mad_dd_ticklabels)
        x_freq = get_x_freq(daily_returns_df)
        date_ticks = pd.date_range(daily_returns_df.index.min(), daily_returns_df.index.max(), freq=x_freq)
        date_labels = date_ticks.strftime('%d %b-%y')
        if x_freq == "M":
            date_labels = date_ticks.strftime('%b-%y')
        if x_freq == "Y":
            date_labels = date_ticks.strftime('%Y')

        plt.yticks(yticks_values, yticks_labels)
        plt.xticks(date_ticks, date_labels)
        plt.axhline(max_drawdown, color='red', linestyle='--', label='Max Drawdown')
        plt.axhline(0, color='black', linestyle='--', alpha=0.5)
        plt.xticks(rotation=25, fontsize=15)
        plt.yticks(fontsize=15)

        plt.legend(loc='lower left', bbox_to_anchor=(0, 0.1), fontsize=15)
        plt.xlabel("Date", fontsize=15)
        plt.ylabel("Drawdown(%)", fontsize=15)
        plt.title("Drawdown", fontsize=15)
        plt.grid(True, alpha=0.2)
        plt.savefig(os.path.join(self.strat_dir, "dd_curve.png"))
        plt.close()

    def plot_pnl_curve(self):
        daily_returns_df = self.daily_returns.copy()
        # ROC Cummulative
        daily_returns_df["pnl_pct_cumulative"] = daily_returns_df["pnl_pct"].cumsum()

        daily_returns_df.set_index("date", inplace=True)
        plt.figure(figsize=(15, 10))
        plt.fill_between(daily_returns_df.index, daily_returns_df["pnl_pct_cumulative"], color='green', alpha=0.5)
        plt.fill_between(daily_returns_df.index, 0, daily_returns_df["pnl_pct_cumulative"],
                         where=(daily_returns_df["pnl_pct_cumulative"] < 0), color='red', alpha=0.7)

        fmt = '%.0f%%'
        yticks = mtick.FormatStrFormatter(fmt)
        plt.gca().yaxis.set_major_formatter(yticks)

        max_pnl_pct_cumulative = daily_returns_df['pnl_pct_cumulative'].max()

        # Set y-axis ticks
        yticks_values = [max_pnl_pct_cumulative, 0]
        yticks_labels = [f"{round(max_pnl_pct_cumulative, 2)}%", "0"]

        # add ticks on Y at interval of max_pnl_pct_cumulative/6
        max_pnl = int(max_pnl_pct_cumulative)
        start_pnl_intrvl = max_pnl // 6
        if start_pnl_intrvl != 0:
            pct_cumulative_ticks = [i for i in range(start_pnl_intrvl, max_pnl - start_pnl_intrvl, start_pnl_intrvl)]
            pct_cumulative_ticklabels = [f"{round(x, 2)}%" for x in pct_cumulative_ticks]
            yticks_values.extend(pct_cumulative_ticks)
            yticks_labels.extend(pct_cumulative_ticklabels)

        x_freq = get_x_freq(daily_returns_df)
        date_ticks = pd.date_range(daily_returns_df.index.min(), daily_returns_df.index.max(), freq=x_freq)
        date_labels = date_ticks.strftime('%d %b-%y')
        if x_freq == "M":
            date_labels = date_ticks.strftime('%b-%y')
        if x_freq == "Y":
            date_labels = date_ticks.strftime('%Y')

        plt.yticks(yticks_values, yticks_labels)
        plt.xticks(date_ticks, date_labels)

        plt.axhline(0, color='black', linestyle='--', alpha=0.5)
        plt.axhline(max_pnl_pct_cumulative, color='green', linestyle='--', label='Max PnL(%)')

        plt.xticks(rotation=25, fontsize=15)
        plt.yticks(fontsize=15)

        # plt.legend(loc="center left")
        plt.legend(loc='upper left', bbox_to_anchor=(0, 0.9), fontsize=15)
        plt.xlabel("Date", fontsize=15)
        plt.ylabel("P&L(%)", fontsize=15)
        plt.title("P&L", fontsize=15)
        plt.grid(True, alpha=0.2)
        plt.savefig(os.path.join(self.strat_dir, "pnl_curve.png"))
        plt.close()

    def drawdown_series(self):
        """Calculates the maximum drawdown"""
        df = self.daily_pnl.copy()
        df.at[0, "pnl"] = self.capital + df.at[0, "pnl"]
        df.set_index("date", inplace=True)
        cumulative_pnl = np.cumsum(df['pnl'])
        max_cumulative_pnl = np.maximum.accumulate(cumulative_pnl)
        drawdown = max_cumulative_pnl - cumulative_pnl
        return drawdown, max_cumulative_pnl

    def drawdown_details(self):
        drawdown, max_cumulative_points = self.drawdown_series()
        drawdown = (drawdown / max_cumulative_points) * 100
        no_dd = drawdown == 0

        # extract dd start dates
        starts = ~no_dd & no_dd.shift(1)
        starts = list(starts[starts].index)

        # extract end dates
        ends = no_dd & (~no_dd).shift(1)
        ends = list(ends[ends].index)

        # no drawdown :)
        if not starts:
            return pd.DataFrame(index=[], columns=['Started', 'Recovered', 'Drawdown', 'Days'])

        # drawdown series begins in a drawdown
        if ends and starts[0] > ends[0]:
            starts.insert(0, drawdown.index[0])

        # series ends in a drawdown fill with last date
        if not ends or starts[-1] > ends[-1]:
            ends.append(drawdown.index[-1])

        # build dataframe from results
        data = []

        for i, _ in enumerate(starts):
            dd = drawdown[starts[i]:ends[i]]
            data.append((starts[i].date(), ends[i].date(), round(dd.max(), 2), (ends[i] - starts[i]).days))

        df = pd.DataFrame(data=data, columns=['Started', 'Recovered', 'Drawdown', 'Days'])
        df.sort_values(by="Drawdown", ascending=False, inplace=True)
        df.reset_index(drop=True, inplace=True)
        if not df.empty:
            if len(df) > 10:
                df = df.head(10)
            df.to_csv(os.path.join(self.strat_dir, "worst_10_drawdowns.csv"), index=False)

    def calculate_metrices(self):
        """Calculates Metrices from tradebook"""
        day_points_df = self.daily_pnl.copy()
        day_points_df.set_index("date", inplace=True)
        day_points_df = day_points_df.dropna()

        # Calculate total trades
        total_trades = len(day_points_df)

        # Calculate total wins
        total_wins = len(day_points_df[day_points_df['pnl'] > 0])

        # Calculate total losses
        total_losses = len(day_points_df[day_points_df['pnl'] < 0])

        # Calculate total points
        total_pnl = np.sum(day_points_df['pnl'])

        # Calculate max win
        max_win = np.max(day_points_df['pnl'])

        # Calculate max loss
        max_loss = np.min(day_points_df['pnl'])

        drawdown, max_cumulative_points = self.drawdown_series()
        max_dd_pct = np.max(drawdown / max_cumulative_points) * 100
        max_dd = np.max(drawdown)

        # Calculate calmar
        calmar = (total_pnl / max_dd) if max_dd > 0 else 0

        # Calculate win rate
        win_rate = (total_wins / total_trades)

        # Calculate RR, avg win points, avg loss points, profit factor
        win_pnl = np.sum(day_points_df[day_points_df['pnl'] > 0]['pnl'])
        loss_pnl = np.sum(day_points_df[day_points_df['pnl'] < 0]['pnl'])
        OA_adj_pnl = (win_pnl - max_win)
        avg_pnl_on_winning_days = np.mean(day_points_df[day_points_df['pnl'] > 0]['pnl'])
        avg_pnl_on_losing_days = np.mean(day_points_df[day_points_df['pnl'] < 0]['pnl'])
        avg_pnl_winning_days_oa_adj = "NA"
        if total_wins != 0:
            avg_pnl_winning_days_oa_adj = round(OA_adj_pnl / total_wins, 1)
        risk_to_reward = round(abs(avg_pnl_on_winning_days / avg_pnl_on_losing_days), 2)
        profit_factor = round(abs(win_pnl / loss_pnl), 2)
        outlier_adjusted_profit_factor = round(abs(OA_adj_pnl / loss_pnl), 2)
        expectancy = (win_rate * risk_to_reward) - (1 - win_rate)

        mean_pnl = np.mean(day_points_df['pnl'])

        # Calculate max loss day
        max_loss_day = day_points_df[day_points_df['pnl'] == max_loss].index[0]

        # Calculate max win day
        max_win_day = day_points_df[day_points_df['pnl'] == max_win].index[0]

        # Calculate sharpe ratio
        sharpe = "NA"
        if np.std(day_points_df['pnl']) != 0:
            sharpe = (np.sqrt(252) * mean_pnl) / np.std(day_points_df['pnl'])

        # Calculate sortino ratio
        down_dev = np.where(day_points_df['pnl'] < 0, day_points_df['pnl'], 0).std()
        sortino = "NA"
        if down_dev != 0:
            sortino = (np.sqrt(252) * mean_pnl) / down_dev

        # Calculate average monthly ROI
        monthly_returns = day_points_df['pnl'].resample('M').sum()
        average_monthly_roi = np.mean(monthly_returns / total_pnl) * 100
        metrics = pd.DataFrame(columns=['Test Start Date', 'Test End Date', 'Instrument',
                                        'Strategy', 'Total Capital', 'Notional Value Asm.', 'Leverage',
                                        'Traded with Lots', 'Total Trading days', 'Win days', 'Loss Days',
                                        'Win Rate (%)', 'Avg Profit on Profit Days Outlier Adjusted (Rs)',
                                        'Avg Profit on Profit Days (Rs)', 'Avg Loss on Loss Days (Rs)',
                                        'Average Monthly ROI (%)', 'Total Profit (Rs)', 'Total Profit (%)',
                                        'Max Profit (Rs)', 'Max Loss (Rs)', 'Max Winning Day', 'Max Losing Day',
                                        'Max Drawdown (Rs)', 'Max Drawdown (%)', 'Risk to reward', 'Profit Factor',
                                        'Outlier Adjusted Profit Factor', 'Expectancy', 'Calmar',
                                        'Sharpe Ratio (Annualised)', 'Sortino Ratio (Annualised)',
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
            'Avg Profit on Profit Days Outlier Adjusted (Rs)': round(avg_pnl_winning_days_oa_adj, 2) if avg_pnl_winning_days_oa_adj != "NA" else avg_pnl_winning_days_oa_adj,
            'Avg Profit on Profit Days (Rs)': round(avg_pnl_on_winning_days, 2),
            'Avg Loss on Loss Days (Rs)': round(avg_pnl_on_losing_days, 2),
            'Average Monthly ROI (%)': round(average_monthly_roi, 2),
            'Total Profit (Rs)': round(total_pnl, 2),
            'Total Profit (%)': round((total_pnl / self.capital) * 100, 2),
            'Max Profit (Rs)': round(max_win, 2),
            'Max Loss (Rs)': round(max_loss, 2),
            'Max Winning Day': max_win_day.date(),
            'Max Losing Day': max_loss_day.date(),
            'Max Drawdown (Rs)': round(max_dd, 2),
            'Max Drawdown (%)': round(max_dd_pct, 2),
            'Risk to reward': risk_to_reward,
            'Profit Factor': profit_factor,
            'Outlier Adjusted Profit Factor': outlier_adjusted_profit_factor,
            'Expectancy': round(expectancy, 2),
            'Calmar': round(calmar, 2),
            'Sharpe Ratio (Annualised)': round(sharpe, 2) if sharpe != "NA" else sharpe,
            'Sortino Ratio (Annualised)': round(sortino, 2) if sortino != "NA" else sortino,
            'Unable to trade days': self.unable_to_trade_days}, index=[0])], ignore_index=True)
        metrics.reset_index(drop=True, inplace=True)
        metrics = metrics.set_index('Test Start Date').T
        metrics.to_csv(self.metrics_path, index_label='Test Start Date')

    def prepare_pdf_report(self):
        """Prepares PDF report"""
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=16)

        title_width = pdf.get_string_width("Backtest Report")
        x_pos = (pdf.w - title_width) / 2

        pdf.set_xy(x_pos, 5)
        pdf.cell(title_width, 10, "Backtest Report", 0, 2, "C")
        pdf.line(x_pos, 15, x_pos + title_width, 15)

        pdf.image(os.path.join(self.strat_dir, "pnl_curve.png"), x=-0.5, y=20, w=(pdf.w + 1) / 1)
        pdf.image(os.path.join(self.strat_dir, "dd_curve.png"), x=-0.5, y=160, w=(pdf.w + 1) / 1)

        pdf.add_page()

        # Add the CSV data to the PDF as a table
        with open(os.path.join(self.strat_dir, "metrics.csv"), "r") as csv_file:
            reader = csv.reader(csv_file)
            data = []
            for row in reader:
                data.append(row)
            title_width = pdf.get_string_width("Key Performance Metrics")
            x_pos = (pdf.w - title_width) / 2
            pdf.set_font("Arial", size=13)
            pdf.set_xy(x_pos, 5)
            pdf.cell(title_width, 10, "Key Performance Metrics", 0, 2, "C")

            pdf.set_font("Arial", size=11)

            col_width = pdf.w / 3
            row_height = pdf.font_size * 2

            for row in data:
                pdf.set_x(col_width / 2 - 18)
                for item in row:
                    pdf.cell(col_width + 20, row_height, str(item), border=1)
                pdf.ln(row_height)
        pdf.add_page()
        try:
            with open(os.path.join(self.strat_dir, "worst_10_drawdowns.csv"), "r") as csv_file:
                reader = csv.reader(csv_file)
                data = []
                for row in reader:
                    data.append(row)
                title_width = pdf.get_string_width("Worst 10 Drawdowns")
                x_pos = (pdf.w - title_width) / 2
                pdf.set_font("Arial", size=12)
                pdf.set_xy(x_pos, 5)
                pdf.cell(title_width, 10, "Worst 10 Drawdowns", 0, 2, "C")

                pdf.set_font("Arial", size=10)

                col_width = pdf.w / 4
                row_height = 10

                for row in data:
                    pdf.set_x(col_width / 2)
                    for item in row:
                        pdf.cell(40, row_height, str(item), border=1)
                    pdf.ln(row_height)
        except Exception as e:
            pass
        try:
            with open(os.path.join(self.strat_dir, "daily_summary.csv"), "r") as summary_file:
                reader = csv.reader(summary_file)
                data = []
                for row in reader:
                    data.append(row)
                daily_summary_width = pdf.get_string_width("Daily Summary")
                x_pos = (pdf.w - daily_summary_width) / 2
                pdf.set_font("Arial", size=12)
                pdf.set_xy(x_pos, 130)
                pdf.cell(daily_summary_width, 10, "Daily Summary", 0, 2, "C")
                pdf.set_font("Arial", size=10)
                col_width = pdf.w / 5
                row_height = 8
                for row in data:
                    pdf.set_x(col_width)
                    for item in row:
                        pdf.cell(25, row_height, str(item), border=1)
                    pdf.ln(row_height)
        except Exception:
            pass
        pdf.add_page()
        pdf.image(os.path.join(self.strat_dir, "monthly_returns.png"), x=4, y=30, w=(pdf.w - 10))

        pdf.output(os.path.join(self.strat_dir, f"{self.strat_name.lower()}_report.pdf"))

    def calculate_daily_summary(self):
        df = self.daily_returns.copy()
        df['day'] = df['date'].dt.strftime('%A')
        daily_summary_df = pd.DataFrame(columns=["Day", "Returns (%)", "Max profit(%)", "Max loss(%)", "Total trades"])
        df_grp = df.groupby("day")
        for day, day_df in df_grp:
            max_profit = round(day_df["pnl_pct"].max(), 2)
            max_loss = round(day_df["pnl_pct"].min(), 2)
            total_return = round(day_df["pnl_pct"].sum(), 2)
            total_trades = day_df["total trade"].sum()
            daily_summary_df = pd.concat([daily_summary_df, pd.DataFrame({"Day": [day], "Returns (%)": [total_return],
                                                                          "Max profit(%)": [max_profit],
                                                                          "Max loss(%)": [max_loss],
                                                                          "Total trades": [total_trades]})],
                                         ignore_index=True)
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
        daily_summary_df['Day'] = pd.Categorical(daily_summary_df['Day'], categories=day_order, ordered=True)
        daily_summary_df = daily_summary_df.sort_values('Day')
        daily_summary_df.to_csv(self.daily_summary_path, index=False)
        print("PDF REPORT GENERATED!!")
