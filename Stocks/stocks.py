w#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 16:01:41 2019

"""
#%matplotlib qt

import json
import urllib.request
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn import linear_model


#%%
# Edelmetallrechner
feinunze = 31.1034768 # Gramm
online_preis_pro_unze_in_USD = 1684.29
EUR_by_USD = 1.082

def preis_in_EUR_pro_oz(oz):
    return oz * online_preis_pro_unze_in_USD / EUR_by_USD


def preis_in_EUR(gramm):
    return gramm / feinunze * online_preis_pro_unze_in_USD / EUR_by_USD


def gold_summary(tatsaechlich):
    for ind, g in enumerate([5, 10, 20]):
        preis = preis_in_EUR(g)
        aufschlag = (tatsaechlich[ind]  - preis) / preis * 100
        out = 'Theoretischer Preis: %.2f EUR pro %i Gramm, ' % (preis_in_EUR(g), g)
        out += 'Ladenpreis: %.2f EUR, Aufschlag %.2f %%' % (tatsaechlich[ind], aufschlag)
        print(out)

tatsaechliche_preise_in_EUR = [285.5, 540.5, 1174.0]
gold_summary(tatsaechliche_preise_in_EUR)


#%%



class Stock:
    
    def __init__(self, name_, symbol_):
        self.name = name_
        self.symbol = symbol_
    
    def set_data(self, data_: pd.DataFrame):
        self.data = data_
        
    def __str__(self):
        return self.name + ' (' + self.symbol + ')'

class Postition:
    
    def __init__(self, stock_: Stock, date_purchased_, price_purchased_, number_):
        self.stock = stock_
        self.date_purchased = date_purchased_
        self.price_purchased = price_purchased_
        self.number = number_
        
    def __str__(self):
        return '{}, {}, {}, {} '.format(
                self.stock.__str__(), self.date_purchased, 
                self.price_purchased, self.number)

class Portfolio:
    
    def __init__(self):
        self.positions = []


class Policy:
    
    def __init__(self, val):
        self.val = val
    
    def A(self, t):
        '''
        make a investment decision given the current portfolio and 
        return the number of
        '''
    
    def __str__(self):
        return "Policy "


class Simulation:
    
    def __init__(self, policy_: Policy):
        self.policy = policy_
    
    def run(self, stock: pd.DataFrame):
        print('test')


#%%

        # 'hh': 'HHFA.DE',
          #  'linde':'LIN.DE'




        
def load(symbol):
    url = "https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={}&apikey=46MMSGHA8VICBAWQ&outputsize=full"
    link = url.format(symbol)
    with urllib.request.urlopen(link) as url:
        data = json.loads(url.read().decode())
    return pd.DataFrame.from_dict(data['Time Series (Daily)'])

def prepare(df):
    ''' prepare the downloaded dat '''
    df = df.T # transpose
    df = df.iloc[::-1] # reverse
    df.columns = ['open', 'high', 'low', 'close', 'volume']
    for c in df.columns:
        df[c] = pd.to_numeric(df[c])
    # moving averages
    df['mv-50'] = df.open.rolling(window=24*30).mean()
    # direction from open to close
    df['direction'] = df.apply(lambda x: 'up' if x['open'] < x['close'] else 'down', 
      axis=1)
    return df
    
       


#%%
        
stocks = {}
ball = Stock('ballard', 'BLDP.TO')
stocks['ballard'] = ball
df = load(stocks['ballard'].symbol)
df = prepare(df)
ball.set_data(df)
print(ball)

#%%

pos1 = Postition(ball, datetime(2019, 11, 30), 5, 200)        
print(pos1)
        
#%%
        

#%%

def gains(df, ndays):
    s = pd.Series(np.repeat(np.nan, df.shape[0]), 
                  name='{}-day-gain'.format(ndays),
                  index=df.index)
    for i in range(ndays, df.shape[0]):
        s[i] = df.open[i] - df.open[i-ndays]
    return pd.concat([df, s], axis=1)

def targets(df):
    ''' add target attributes '''
    df = gains(df,1)
    df = gains(df,20)
    return df

df = targets(df)

#%%

# plot
ax = plt.gca()
df.reset_index().plot(kind='line', style='.', markersize=1, x='index', y='open', ax=ax)
df.reset_index().plot(kind='line', x='index', y='mv-50', color='green', ax=ax)
df.reset_index().plot(kind='bar', x='index', y='20-day-gain', color='yellow', ax=ax)
plt.title('Some stock')
plt.ylabel('price')


# linear regression 
period = 30
df1 = df.iloc[df.shape[0]-period: , :].copy()
X = np.arange(df1.shape[0]) 
X = X.reshape(df1.shape[0], 1) # turn into 2D
Y = df1.open.values.reshape(df1.shape[0], 1)
regr = linear_model.LinearRegression() 
regr.fit(X, Y)
df1['lin-reg'] = regr.predict(X)
df2 = pd.concat([df, df1['lin-reg']], axis=1, join='outer') # add to df


# plot
ax = plt.gca()
df2.reset_index().plot(kind='line', style='.', markersize=2, x='index', y='open', ax=ax)
df2.reset_index().plot(kind='line', style='-', linewidth=1, x='index', y='mv-50', ax=ax)
df2.reset_index().plot(kind='line', style='-', linewidth=3, x='index', y='lin-reg', ax=ax)

# plotly does not work in spyder

# ohlc

#%%