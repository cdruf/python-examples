import pandas as pd
import plotly.graph_objects as go

df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/finance-charts-apple.csv')
fig = go.Figure(data=go.Ohlc(x=df['Date'],
                             open=df['AAPL.Open'],
                             high=df['AAPL.High'],
                             low=df['AAPL.Low'],
                             close=df['AAPL.Close']))
fig.show()
