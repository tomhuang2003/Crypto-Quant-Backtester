import yfinance as yf



df = yf.download("BTC-USD", period="max")


#*********************
#   BTC/hold MA5/15
#*********************
df['MA_Short'] = df['Close'].rolling(2).mean()
df['MA_Long'] = df['Close'].rolling(35).mean()

df['Signal'] = (df['MA_Short'] > df['MA_Long']).astype(int)

df['Daily_Return'] = df['Close'].pct_change()
df['Strategy_Return'] = df['Signal'].shift(1) * df['Daily_Return']

total_profit = (1 + df['Strategy_Return'].fillna(0)).cumprod().iloc[-1] - 1
buy_and_hold = (1 + df['Daily_Return'].fillna(0)).cumprod().iloc[-1] - 1


#********************************
# Transaction fee and slippage
#********************************
df['Trade'] = df['Signal'].diff().fillna(0).abs()

fee = 0.0015 #Transaction fee 0.1%,slippage 0.05%
df['Strategy_Return_Net'] = df['Strategy_Return'] - (df['Trade'] * fee)
net_total_profit = (1 + df['Strategy_Return_Net'].fillna(0)).cumprod().iloc[-1] - 1


print(f"📊 原始策略收益: {total_profit * 100:.2f}%")
print(f"💸 扣除手續費後收益: {net_total_profit * 100:.2f}%")
print(f"🔄 總交易次數: {df['Trade'].sum()} 次")
print(f"💰 純持有 BTC 收益: {buy_and_hold * 100:.2f}%",end="\n\n")


#****************************
#   btc/hold max drawdown
#****************************
equity_curve = (1 + df['Strategy_Return_Net'].fillna(0)).cumprod()
rolling_max = equity_curve.rolling(window=len(df), min_periods=1).max()
drawdown = (equity_curve - rolling_max) / rolling_max
max_drawdown = drawdown.min()

print(f"📉 策略的最大回撤 (MDD): {max_drawdown * 100:.2f}%")

market_equity = (1 + df['Daily_Return'].fillna(0)).cumprod()
market_rolling_max = market_equity.rolling(window=len(df), min_periods=1).max()
market_drawdown = (market_equity - market_rolling_max) / market_rolling_max
market_mdd = market_drawdown.min()

print(f"📉 BTC 純持有的最大回撤 (Market MDD): {market_mdd * 100:.2f}%")


#************************
#     Sharpe Ratio
#************************

# Assuming 365 trading days for Crypto
sharpe_ratio = (df['Strategy_Return_Net'].mean() / df['Strategy_Return_Net'].std()) * (365**0.5)
print(f"📊 Sharpe Ratio: {sharpe_ratio:.2f}")






