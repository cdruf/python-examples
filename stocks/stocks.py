#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 16:01:41 2019

"""

import json
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import linear_model


@dataclass
class Stock:
    name: str
    symbol: str
    data: pd.DataFrame = field(init=False)

    def __str__(self):
        return self.name + ' (' + self.symbol + ')'


@dataclass
class Position:
    stock: Stock
    date_purchased: datetime
    price_purchased: float
    number: int

    def __str__(self):
        return f"{self.stock}, {self.date_purchased}, {self.price_purchased}, {self.number}"


class Portfolio:

    def __init__(self):
        self.positions = []


class Policy:

    def __init__(self, val):
        self.val = val

    def A(self, t):
        """
        Make an investment decision given the current portfolio and
        return the number of
        """

    def __str__(self):
        return "Policy"


class Simulation:

    def __init__(self, policy_: Policy):
        self.policy = policy_

    def run(self, stock: pd.DataFrame):
        print('test')


# 'hh': 'HHFA.DE',
#  'linde':'LIN.DE'


def load(symbol):
    url = "https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={}&apikey=46MMSGHA8VICBAWQ&outputsize=full"
    link = url.format(symbol)
    with urllib.request.urlopen(link) as url:
        data = json.loads(url.read().decode())
    return pd.DataFrame.from_dict(data['Time Series (Daily)'])


def prepare(df):
    """Prepare the downloaded data"""
    df = df.T  # transpose
    df = df.iloc[::-1]  # reverse
    df.columns = ['open', 'high', 'low', 'close', 'volume']
    for c in df.columns:
        df[c] = pd.to_numeric(df[c])
    # moving averages
    df['mv-50'] = df.open.rolling(window=24 * 30).mean()
    # direction from open to close
    df['direction'] = df.apply(lambda x: 'up' if x['open'] < x['close'] else 'down',
                               axis=1)
    return df


def gains(df, ndays):
    s = pd.Series(np.repeat(np.nan, df.shape[0]),
                  name='{}-day-gain'.format(ndays),
                  index=df.index)
    for i in range(ndays, df.shape[0]):
        s[i] = df.open[i] - df.open[i - ndays]
    return pd.concat([df, s], axis=1)


def targets(df):
    """ add target attributes """
    df = gains(df, 1)
    df = gains(df, 20)
    return df


if __name__ == '__main__':
    stocks = {}
    ball = Stock('ballard', 'BLDP.TO')
    stocks['ballard'] = ball
    df = load(stocks['ballard'].symbol)
    df = prepare(df)
    ball.data = df
    print(ball)

    pos1 = Position(ball, datetime(2019, 11, 30), 5, 200)
    print(pos1)

    df = targets(df)

# %%

# plot
ax = plt.gca()
df.reset_index().plot(kind='line', style='.', markersize=1, x='index', y='open', ax=ax)
df.reset_index().plot(kind='line', x='index', y='mv-50', color='green', ax=ax)
df.reset_index().plot(kind='bar', x='index', y='20-day-gain', color='yellow', ax=ax)
plt.title('Some stock')
plt.ylabel('price')

# linear regression
period = 30
df1 = df.iloc[df.shape[0] - period:, :].copy()
X = np.arange(df1.shape[0])
X = X.reshape(df1.shape[0], 1)  # turn into 2D
Y = df1.open.values.reshape(df1.shape[0], 1)
regr = linear_model.LinearRegression()
regr.fit(X, Y)
df1['lin-reg'] = regr.predict(X)
df2 = pd.concat([df, df1['lin-reg']], axis=1, join='outer')  # add to df

# plot
ax = plt.gca()
df2.reset_index().plot(kind='line', style='.', markersize=2, x='index', y='open', ax=ax)
df2.reset_index().plot(kind='line', style='-', linewidth=1, x='index', y='mv-50', ax=ax)
df2.reset_index().plot(kind='line', style='-', linewidth=3, x='index', y='lin-reg', ax=ax)
