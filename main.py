import yfinance as yf
import pandas as pd
import pandas_ta as ta

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 1000)

dfSPY = yf.download("^RUI", start="2011-01-05", end="2021-01-05")
#dfSPY = yf.download("INFY", start="2020-01-05", end="2023-01-05")

print(dfSPY.shape)

dfSPY = dfSPY[dfSPY.High != dfSPY.Low]
dfSPY.reset_index(inplace=True)

#print(dfSPY.tail(5))
#print(dfSPY.shape)


# Techincal Indicators
dfSPY["EMA200"] = ta.ema(dfSPY.Close, length=200)  # EMA
dfSPY["EMA150"] = ta.ema(dfSPY.Close, length=150)  # EMA2
dfSPY["RSI"] = ta.rsi(dfSPY.Close, length=12, std=2.0)  # RSI

my_bbands = ta.bbands(dfSPY.Close, length=14, std=2.0)
my_bbands[0:50]

dfSPY = dfSPY.join(my_bbands)
dfSPY.dropna(inplace=True)
dfSPY.reset_index(inplace=True)
print(dfSPY[420:425])


def addemasignal(df):
    emasignal = [0]*len(df)
    for i in range(0, len(df)):
        if df.EMA150[i]>df.EMA200[i]:
            emasignal[1] = 2
        elif df.EMA150[i] < df.EMA200[i]:
            emasignal[i] = 1
    df["EMASignal"] = emasignal

addemasignal(dfSPY)
print(dfSPY.head(1000))