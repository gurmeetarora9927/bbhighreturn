import yfinance as yf
import pandas as pd

dfSPY = yf.download("^RUI", start="2011-01-05", end="2021-01-05")

print(dfSPY.tail(5))
print(dfSPY.shape)
