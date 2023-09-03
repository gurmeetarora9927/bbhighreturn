#!pip install pandas_ta
#!pip install plotly
#!pip install backtesting
#!pip install bokeh==3.1.1

import numpy as np
import pandas as pd
import yfinance as yf
import pandas_ta as ta
import backtesting as bt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 1000)
bt.set_bokeh_output(notebook=False)


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
dfSPY["RSI"] = ta.rsi(dfSPY.Close, length=12)  # RSI

my_bbands = ta.bbands(dfSPY.Close, length=14, std=2.0)
my_bbands[0:50]

dfSPY = dfSPY.join(my_bbands)
dfSPY.dropna(inplace=True)
dfSPY.reset_index(inplace=True)
#print(dfSPY[420:425])


# This is the trend signal. 1 = DownTrend  , 2 = Uptrend
def addemasignal(df):
    emasignal = [0]*len(df)
    for i in range(0, len(df)):
        if df.EMA150[i]>df.EMA200[i]:
            emasignal[1] = 2  # Uptrend
        elif df.EMA150[i] < df.EMA200[i]:
            emasignal[i] = 1  # Downtrend
    df["EMASignal"] = emasignal

addemasignal(dfSPY)
#print(dfSPY.head(1000))

# Order Signal
def addorderslimit(df, percent):
    ordersignal = [0]*len(df)
    for i in range(1, len(df)):
        # SELL Signal. Close price more than upper band and trend=Downtrend
        if df.Close[i] <= df['BBL_14_2.0'][i] and df.EMASignal[i] == 2:
            ordersignal[i] = df.Close[i] - df.Close[i]*percent
        # BUY Signal. Close price less than lower band and trend=Uptrend
        if df.Close[i] >= df['BBU_14_2.0'][i] and df.EMASignal[i] == 1:
            ordersignal[i] = df.Close[i]*(1+percent)
    df['ordersignal'] = ordersignal

addorderslimit(dfSPY, 0.0)
#print(dfSPY[dfSPY.ordersignal != 0.0])


# Visualization
# Point POS Break
def pointposbreak(x):
    if x['ordersignal'] != 0:
        return x['ordersignal']
    else:
        return np.nan
dfSPY['pointposbreak'] = dfSPY.apply(lambda row: pointposbreak(row), axis=1)


dfpl = dfSPY[1000:3200].copy()
fig = go.Figure(data=[go.Candlestick(x=dfpl.index,
                                     open=dfpl['Open'],
                                     high=dfpl['High'],
                                     low=dfpl['Low'],
                                     close=dfpl['Close']),
                      go.Scatter(x=dfpl.index, y=dfpl.EMA200, line=dict(color='orange', width=2), name="EMA200"),
                      go.Scatter(x=dfpl.index, y=dfpl.EMA150, line=dict(color='yellow', width=2), name="EMA150"),
                      go.Scatter(x=dfpl.index, y=dfpl['BBL_14_2.0'], line=dict(color='blue', width=1), name="BBLow"),
                      go.Scatter(x=dfpl.index, y=dfpl['BBU_14_2.0'], line=dict(color='blue', width=1), name="BBUp")])

fig.add_scatter(x=dfpl.index, y=dfpl['pointposbreak'], mode="markers",
                marker=dict(size=6, color="MediumPurple"), name="Signal")
fig.update_xaxes(rangeslider_visible=False)
fig.update_layout(autosize=False, width=600, height=600, margin=dict(l=50, r=30, b=100, t=100, pad=4), paper_bgcolor="black")
#fig.show()

dfpl = dfSPY[:].copy()
def SIGNAL():
    return dfpl.ordersignal

from backtesting import Strategy
from backtesting import Backtest

class MyStrat(Strategy):
    initsize=0.99
    mysize=initsize

    def init(self):
        super().init()
        self.signal = self.I(SIGNAL)


    def next(self):
        super().next()
        TPSLRatio = 2
        perc = 0.02

        if len(self.trades) > 0:
            if self.data.index[-1] - self.trades[-1].entry_time >= 10:
                self.trades[-1].close()
            if self.trades[-1].is_long and self.data.RSI[-1] >= 75:
                self.trades[-1].close()
            elif self.trades[-1].is_short and self.data.RSI[-1] <= 25:
                self.trades[-1].close()

        if self.signal != 0 and len(self.trades) == 0 and self.data.EMASignal == 2:
            sl1 = min(self.data.Low[-1], self.data.Low[-2]) * (1-perc)
            tp1 = self.data.Close[-1] + (self.data.Close[-1] - sl1)*TPSLRatio
            self.sell(sl=sl1, tp=tp1, size=self.mysize)

        if self.signal != 0 and len(self.trades) == 0 and self.data.EMASignal == 1:
            sl1 = max(self.data.High[-1], self.data.High[-2]) * (1+perc)
            tp1 = self.data.Close[-1] - (sl1 - self.data.Close[-1])*TPSLRatio
            self.sell(sl=sl1, tp=tp1, size=self.mysize)


bt = Backtest(dfpl, MyStrat, cash=1000, margin=1/5, commission=0.000)
stat = bt.run()
print(stat)

#bt.plot()


























