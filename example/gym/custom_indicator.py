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
import numpy as np


# def NormalizedScore(df: DataFrame, length=30):
#     ll = df['low'].rolling(window=length).min()
#     hh = df['high'].rolling(window=length).max()
#     vll = df['volume'].rolling(window=length).min()
#     vhh = df['volume'].rolling(window=length).max()
#     scale = hh - ll
#     vscale = vhh - vll
#
#     df['feature_normal_open'] = (df['open'] - ll)/scale
#     df['feature_normal_low'] = (df['low'] - ll) / scale
#     df['feature_normal_high'] = (df['high'] - ll) / scale
#     df['feature_normal_close'] = (df['close'] - ll) / scale
#     df['feature_normal_volume'] = (df['volume']-vll)/vscale
#     return


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


def SMI(df: DataFrame, k_length=9, d_length=3):
    """
    The Stochastic Momentum Index (SMI) Indicator was developed by
    William Blau in 1993 and is considered to be a momentum indicator
    that can help identify trend reversal points

    :return: DataFrame with smi column populated
    """

    ll = df['low'].rolling(window=k_length).min()
    hh = df['high'].rolling(window=k_length).max()

    diff = hh - ll
    rdiff = df['close'] - (hh + ll) / 2

    avgrel = rdiff.ewm(span=d_length).mean().ewm(span=d_length).mean()
    avgdiff = diff.ewm(span=d_length).mean().ewm(span=d_length).mean()

    df['feature_smi'] = np.where(avgdiff != 0, avgrel / (avgdiff / 2), 0)

    return df


if __name__ == '__main__':
    df = pd.read_pickle("./data/raw/binance-BTCUSDT-5m.pkl")
    NormalizedScore(df)
    print(df)