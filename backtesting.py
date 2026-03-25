import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class crypto_data:
    def __init__(self,ticker,period = "max"):#start = ,end= #period="max"
        self.ticker = ticker
        self.period = period
        self.df = None

    def fetch_data(self):
        print(f"🚀🚀loading  data...🚀🚀")
        self.df = yf.download(tickers=self.ticker, period = self.period)
        if isinstance(self.df.columns, pd.MultiIndex):
            self.df.columns = self.df.columns.get_level_values(0)
        self.df['Close'] = self.df['Close'].ffill()

        self.df = self.df.copy()
        return self.df

    def calculate_indicators(self):
        close = self.df['Close'].astype(float)
        high = self.df['High'].astype(float)
        low = self.df['Low'].astype(float)

        ######  Wilder's RSI  ######
        delta = close.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.ewm(alpha=1 / 14, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1 / 14, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        self.df['RSI'] = 100 - (100 / (1 + rs))


        ######  ATR  ######
        hl = high - low
        hc = (high - close.shift()).abs()
        lc = (low - close.shift()).abs()
        tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
        self.df['ATR'] = tr.ewm(alpha=1 / 14, adjust=False).mean()
        self.df['ATR_Stop_Pct'] = -(self.df['ATR'].values * 2.5 / close.values)
        self.df['Daily_Return'] = close.pct_change()

        self.df.dropna(inplace=True)


        return self.df

    def run_backtesting(self,short_range, long_range, rsi_thresh_range):
        print("⚙️⚙️Backtesting...⚙️⚙️")
        all_results = []
        fee_and_slippage = 0.003
        hard_stop = -0.1

        for thresh in rsi_thresh_range:
            for s in short_range:
                for l in long_range:
                    ma_s = self.df['Close'].rolling(s).mean()
                    ma_l = self.df['Close'].rolling(l).mean()

                    ######  trade signal  ######
                    signal = ((ma_s > ma_l) & (self.df['RSI'] > thresh)).astype(int)
                    trade = signal.diff().fillna(0).abs()

                    ######  Dual-Stop Mechanism  ######
                    raw_ret = signal.shift(1) * self.df['Daily_Return']
                    dynamic_stop = self.df['ATR_Stop_Pct'].shift(1)
                    effective_stop = np.maximum(hard_stop, dynamic_stop)

                    is_stop = (signal.shift(1) == 1) & (self.df['Daily_Return'] < effective_stop)
                    protected_ret = np.where(is_stop, effective_stop, raw_ret)
                    net_daily_ret = (1 + protected_ret) * (1 - trade * fee_and_slippage) - 1

                    equity_curve = (1 + net_daily_ret.fillna(0)).cumprod()
                    total_ret = float(equity_curve.iloc[-1] - 1)

                    ######  maximum drawdown  ######
                    rolling_max = equity_curve.rolling(window=len(self.df), min_periods=1).max()
                    mdd = float(((equity_curve - rolling_max) / rolling_max).min())

                    ######  Sharpe & Sortino  ######
                    curr_mean = float(net_daily_ret.mean())
                    curr_std = float(net_daily_ret.std())
                    sharpe = (curr_mean / curr_std) * (365 ** 0.5) if curr_std > 0 else 0

                    downside_std = net_daily_ret[net_daily_ret < 0].std()
                    sortino = (curr_mean / downside_std) * (365 ** 0.5) if downside_std > 0 else 0


                    ######  Report  ######
                    active_trades = net_daily_ret[net_daily_ret != 0]
                    win_rate = (active_trades > 0).mean() if len(active_trades) > 0 else 0
                    avg_win = active_trades[active_trades > 0].mean() if win_rate > 0 else 0
                    avg_loss = active_trades[active_trades < 0].abs().mean() if win_rate < 1 else 0
                    p_l_ratio = avg_win / avg_loss if avg_loss != 0 else 0
                    expectancy = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)

                    all_results.append({
                        'RSI_Thresh': thresh,
                        'Short': s,
                        'Long': l,
                        'Profit_%': round(total_ret * 100, 1),
                        'Sharpe': round(float(sharpe), 2),
                        'Sortino': round(float(sortino), 2),
                        'MDD_%': round(mdd * 100, 1),
                        'Trades': int(trade.sum()),
                        'Win_Rate_%': round(win_rate * 100, 2),
                        'PL_Ratio': round(p_l_ratio, 2),
                        'Expectancy_%': round(expectancy * 100, 4),
                        'Max_Daily_Loss_%': round(active_trades.min() * 100, 2) if len(active_trades) > 0 else 0
                    })
        self.results_df = pd.DataFrame(all_results)
        return self.results_df


    def get_report(self, n=10):
            if self.results_df is None:
                return "❌ Can't find the backtesting data, run_backtesting() first"

            top_n = self.results_df.sort_values(by='Sharpe', ascending=False).head(n)
            best = top_n.iloc[0]

            plt.figure(figsize=(12, 6))

            ###  plot graph  ###
            bh_curve = (1 + self.df['Daily_Return'].fillna(0)).cumprod()

            s, l, thresh = int(best['Short']), int(best['Long']), best['RSI_Thresh']
            ma_s = self.df['Close'].rolling(s).mean()
            ma_l = self.df['Close'].rolling(l).mean()
            sig = ((ma_s > ma_l) & (self.df['RSI'] > thresh)).astype(int)
            strat_ret = sig.shift(1) * self.df['Daily_Return']
            strat_curve = (1 + strat_ret.fillna(0)).cumprod()

            plt.plot(strat_curve, label=f'Strategy (Sharpe: {best["Sharpe"]})', color='blue', linewidth=2)
            plt.plot(bh_curve, label='Buy & Hold (BTC-USD)', color='gray', linestyle='--', alpha=0.6)

            plt.title(f'Backtesting Result: {self.ticker}', fontsize=14)
            plt.xlabel('Date')
            plt.ylabel('Cumulative Return')
            plt.legend()
            plt.grid(True, alpha=0.3)

            plt.savefig('equity_curve.png')
            print("\n📈 Equity curve saved as 'equity_curve.png'")
            plt.show()




            buy_and_hold_ret = (1 + self.df['Daily_Return'].fillna(0)).cumprod().iloc[-1] - 1


            print("\n" + "★" * 50)
            print(f"📊 {self.ticker} Backtesting Results")
            print(f"💰 Total profit of b/h at the same preiod: {buy_and_hold_ret * 100:.2f}%")
            print("★" * 50)
            print(f"🎯 Best indicator (RSI:{int(best['RSI_Thresh'])}, S:{int(best['Short'])}, L:{int(best['Long'])})")
            print(f"   - Win rate: {best['Win_Rate_%']}%")
            print(f"   - P/L ratio: {best['PL_Ratio']}")
            print(f"   - Expectancy: {best['Expectancy_%']}% / 每筆交易")
            print(f"   - Max daily loss: {best['Max_Daily_Loss_%']}%")

            W = best['Win_Rate_%'] / 100
            R = best['PL_Ratio']
            kelly_f = W - ((1 - W) / R) if R > 0 else 0
            print(f"💰 Suggest (Half-Kelly): {max(0, kelly_f * 0.5) * 100:.1f}%")

            return top_n.drop(columns=['Win_Rate_%', 'PL_Ratio', 'Expectancy_%', 'Max_Daily_Loss_%'])



btc_engine = crypto_data("BTC-USD",period = "max")#period="max"
btc_engine.fetch_data()
btc_engine.calculate_indicators()


#########################################################################
#########################################################################
btc_engine.run_backtesting(
 short_range=range(5, 10),
 long_range=range(30, 60),
 rsi_thresh_range=range(50, 58)
)
report = btc_engine.get_report(n=10)
print(report.to_string(index=False))
#########################################################################
#########################################################################




def run_wfa(engine,start_year, end_year, train_size=2, test_size=1):
    all_oos_results = []
    df_all = engine.fetch_data()
    oos_equity_list = []

    for year in range(start_year, end_year - train_size, test_size):
        train_start = f"{year}-01-01"
        train_end = f"{year + train_size - 1}-12-31"
        test_start = f"{year + train_size}-01-01"
        test_end = f"{year + train_size + test_size - 1}-12-31"

        print(f"🔄 Training: {train_start} ~ {train_end} | Testing: {test_start} ~ {test_end}")


        ######  finding the best indicator  ######
        engine.df = df_all.loc[train_start:train_end].copy()
        if engine.df.empty: continue
        engine.calculate_indicators()
        engine.run_backtesting(
            short_range=range(5, 10),
            long_range=range(30, 60),
            rsi_thresh_range=range(50, 58)
        )

        if engine.results_df.empty: continue
        best_params = engine.results_df.sort_values(by='Sharpe', ascending=False).iloc[0]

        ######  OOP testing  ######
        test_df = df_all.loc[test_start:test_end].copy()
        if test_df.empty: break
        engine.df = test_df
        engine.calculate_indicators()

        oos_res = engine.run_backtesting(
            short_range=[int(best_params['Short'])],
            long_range=[int(best_params['Long'])],
            rsi_thresh_range=[int(best_params['RSI_Thresh'])]
        )

        oos_res['Period'] = year + train_size
        all_oos_results.append(oos_res)

    if not all_oos_results:
        return None

    final_report = pd.concat(all_oos_results)
    return final_report


wfa_report = run_wfa(btc_engine, 2017, 2025)
if wfa_report is not None:
    print("\n" + "=" * 50)
    print("🚀 WFA Result (Analyst year by year)")
    print("=" * 50)
    print(wfa_report[['Period', 'RSI_Thresh', 'Short', 'Long', 'Profit_%', 'Sharpe', 'MDD_%']])

    print(f"\n📈 WFA average Sharpe: {wfa_report['Sharpe'].mean():.2f}")
    print(f"📉 WFA worst MDD: {wfa_report['MDD_%'].min()}%")







