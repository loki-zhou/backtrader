import pandas as pd
import numpy as np
import pandas_ta as ta
import legendary_ta as lta

df = pd.read_pickle("./data/raw/binance-BTCUSDT-5m.pkl")
dataframe = lta.breakouts(df)
print(dataframe)
