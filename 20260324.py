import sqlite3
import pandas as pd
import yfinance as yf

conn = sqlite3.connect('trading_data.db')
cursor = conn.cursor()

cursor.execute('''
    CREATE TABLE IF NOT EXISTS btc_price (
        Date TEXT PRIMARY KEY,
        Close REAL,
        Volume INTEGER
    )
''')

df = yf.download("BTC-USD", period="max")
df.columns = df.columns.get_level_values(0)

df_reset = df.reset_index()


df_reset[['Date', 'Close', 'Volume']].to_sql('btc_price', conn, if_exists='replace', index=False)


conn.commit()
print("✅ 數據已成功存入 SQL 數據庫！")


all_data = pd.read_sql_query("SELECT * FROM btc_price ORDER BY Date ASC", conn)


conn.close()


if not all_data.empty:
    print("-" * 30)
    print(f"📊 資料庫總行數: {len(all_data)}")
    print(f"📅 第一筆數據日期: {all_data['Date'].iloc[0]}")
    print(f"📅 最後一筆數據日期: {all_data['Date'].iloc[-1]}")
    print(f"💰 最新收盤價: {all_data['Close'].iloc[-1]:.2f}")


