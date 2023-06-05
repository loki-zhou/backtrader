from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import datetime  # For datetime objects
import os.path  # To manage paths
import sys  # To find out the script name (in argv[0])
# Import the backtrader platform
import backtrader as bt
import numpy as np

import datetime  # For datetime objects
import os.path  # To manage paths
import sys  # To find out the script name (in argv[0])

# Import the backtrader platform
import backtrader as bt
import backtrader.indicators as btind # 导入策略分析模块



class SSLChanelIndicator(bt.Indicator):
    lines = ('ssl_down','ssl_up',)


    def __init__(self):
        test = self.data
        print("init  start    ", "*"*10)
        for alias in self.data.lines.getlinealiases():
            print(alias)
        print("init  end    ", "*" * 10)


    def next(self):
        print("next SSLChanelIndicator ", self.data.close[0])

class MovingAverageSimplev2(btind.MovingAverageBase):
    '''
    Non-weighted average of the last n periods

    Formula:
      - movav = Sum(data, period) / period

    See also:
      - http://en.wikipedia.org/wiki/Moving_average#Simple_moving_average
    '''
    alias = ('SMA', 'SimpleMovingAverage',)
    lines = ('sma','sx',)

    def __init__(self):
        # Before super to ensure mixins (right-hand side in subclassing)
        # can see the assignment operation and operate on the line
        self.lines[0] = btind.Average(self.data, period=self.p.period)

        super(MovingAverageSimplev2, self).__init__()

    def next(self):
        print(len(self.data.lines[0]))
        self.lines[1] = bt.Max(0,len(self.data.lines[0]))




class TestStrategy(bt.Strategy):

    def log(self, txt, dt=None):
        ''' Logging function for this strategy'''
        dt = dt or self.datas[0].datetime.date(0)
        print('%s, %s' % (dt.isoformat(), txt))

    def __init__(self):
        # Keep a reference to the "close" line in the data[0] dataseries
        self.dataclose = self.datas[0].close
        print("TestStrategy  start    ", "*"*10)
        for alias in self.data.lines.getlinealiases():
            print(alias)
        print("TestStrategy  end    ", "*" * 10)
        self.sma1 = MovingAverageSimplev2()


    def next(self):
        # Simply log the closing price of the series from the reference
        #print(self.env.indicators.SSLChanelIndicatorv2)
        self.log('Close, %.2f' % self.dataclose[0])
        print(self.sma1sx)

if __name__ == '__main__':
    # Create a cerebro entity
    cerebro = bt.Cerebro()
    # cerebro.addindicator(SSLChanelIndicator)
    cerebro.addstrategy(TestStrategy)
    # Datas are in a subfolder of the samples. Need to find where the script is
    # because it could have been called from anywhere
    modpath = os.path.dirname(os.path.abspath(sys.argv[0]))
    datapath = os.path.join(modpath, '../../datas/orcl-1995-2014.txt')

    # Create a Data Feed
    data = bt.feeds.YahooFinanceCSVData(
        dataname=datapath,
        # Do not pass values before this date
        fromdate=datetime.datetime(2000, 1, 1),
        # Do not pass values before this date
        todate=datetime.datetime(2000, 12, 31),
        # Do not pass values after this date
        reverse=False)

    # Add the Data Feed to Cerebro
    cerebro.adddata(data)

    #cerebro.addstrategy(TestStrategy)
    print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())
    # Run over everything
    cerebro.run(runonce=False )
    print('Final Portfolio Value: %.2f' % cerebro.broker.getvalue())
