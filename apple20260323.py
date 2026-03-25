import yfinance as yf
import pandas as pd

import yfinance as yf

df = yf.download("AAPL", period="6mo")

df['MA5'] = df['Close'].rolling(5).mean()
df['MA20'] = df['Close'].rolling(20).mean()

df['Signal'] = (df['MA5'] > df['MA20']).astype(int)

df['Daily_Return'] = df['Close'].pct_change()

df['Strategy_Return'] = df['Signal'].shift(1) * df['Daily_Return']

total_profit = (1 + df['Strategy_Return'].fillna(0)).cumprod().iloc[-1] - 1

print(f"策略收益: {total_profit * 100:.2f}%")


















