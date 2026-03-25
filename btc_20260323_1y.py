import yfinance as yf
import matplotlib.pyplot as plt

df = yf.download("BTC-USD", period="1y")

df['MA_Short'] = df['Close'].rolling(5).mean()
df['MA_Long'] = df['Close'].rolling(15).mean()

df['Signal'] = (df['MA_Short'] > df['MA_Long']).astype(int)

df['Daily_Return'] = df['Close'].pct_change()
df['Strategy_Return'] = df['Signal'].shift(1) * df['Daily_Return']

total_profit = (1 + df['Strategy_Return'].fillna(0)).cumprod().iloc[-1] - 1
print(f"🚀 BTC MA 策略總收益: {total_profit * 100:.2f}%")


buy_and_hold = (1 + df['Daily_Return'].fillna(0)).cumprod().iloc[-1] - 1
print(f"💰 純持有 BTC 收益: {buy_and_hold * 100:.2f}%")



# 1. 計算策略的累積淨值 (Equity Curve)
equity_curve = (1 + df['Strategy_Return'].fillna(0)).cumprod()

# 2. 計算歷史最高點 (High Water Mark)
rolling_max = equity_curve.rolling(window=len(df), min_periods=1).max()

# 3. 計算回撤 (Drawdown)：現在的錢比歷史最高點少了幾 %
drawdown = (equity_curve - rolling_max) / rolling_max

# 4. 找出歷史上最慘的一次跌幅 (Maximum Drawdown)
max_drawdown = drawdown.min()

print(f"📉 策略的最大回撤 (MDD): {max_drawdown * 100:.2f}%")

# 計算市場（純持有）的累積淨值
market_equity = (1 + df['Daily_Return'].fillna(0)).cumprod()
# 計算市場的歷史最高點
market_rolling_max = market_equity.rolling(window=len(df), min_periods=1).max()
# 計算市場的回撤
market_drawdown = (market_equity - market_rolling_max) / market_rolling_max
# 算出市場的最大回撤
market_mdd = market_drawdown.min()

print(f"📉 BTC 純持有的最大回撤 (Market MDD): {market_mdd * 100:.2f}%")




strategy_cum = (1 + df['Strategy_Return'].fillna(0)).cumprod()
market_cum = (1 + df['Daily_Return'].fillna(0)).cumprod()

plt.figure(figsize=(12, 6))
plt.plot(strategy_cum, label='My Strategy (5, 15)', color='green')
plt.plot(market_cum, label='BTC Buy & Hold', color='red', alpha=0.5)

plt.title('Strategy vs Market: BTC-USD')
plt.legend()
plt.grid(True)
plt.show()
