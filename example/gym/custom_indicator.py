'''
//@version=2
study("Normalized Candlesticks (Stochastic Candles)",shorttitle="NC_SH",overlay=false)
p = input(50,title="Length")

//scale
hh = highest(high,p)
ll = lowest(low,p)
scale = hh-ll

//dynamic OHLC
dyno = ((open-ll)/scale)*100
dynl = ((low-ll)/scale)*100
dynh = ((high-ll)/scale)*100
dync = ((close-ll)/scale)*100

//candle color
color=close>open?1:0

//drawcandle
hline(78.6)
hline(61.8)
hline(50)
hline(38.2)
hline(23.6)
plotcandle(dyno,dynh,dynl,dync,title="Candle",color=color==1?green:red)
'''
from pandas import DataFrame
import pandas as pd


def NormalizedScore(df: DataFrame, length=30):
    ll = df['low'].rolling(window=length).min()
    hh = df['high'].rolling(window=length).max()
    vll = df['volume'].rolling(window=length).min()
    vhh = df['volume'].rolling(window=length).max()
    scale = hh - ll
    vscale = vhh - vll

    df['feature_normal_open'] = (df['open'] - ll)/scale
    df['feature_normal_low'] = (df['low'] - ll) / scale
    df['feature_normal_high'] = (df['high'] - ll) / scale
    df['feature_normal_close'] = (df['close'] - ll) / scale
    df['feature_normal_volume'] = (df['volume']-vll)/vscale
    return

if __name__ == '__main__':
    df = pd.read_pickle("./data/raw/binance-BTCUSDT-5m.pkl")
    dataframe = NormalizedScore(df)
    print(dataframe)