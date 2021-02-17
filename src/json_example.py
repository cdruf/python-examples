#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 16:08:04 2019

"""

import json

'''
json.load(...)
Deserialize fp (a .read()-supporting text file or binary file containing a JSON document) 
to a Python object using this conversion table.

json.loads(...)
Deserialize s (a str, bytes or bytearray instance containing a JSON document) 
to a Python object using this conversion table.
'''



with open('/home/cpunkt/Dropbox/workspace_spyder/JsonEx/data.json') as f:
    data = json.load(f)

print(data)
type(data) # dict

data["maps"][0]["id"]
data["masks"]["id"]
data["om_points"]



link = "https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=ADS.DE&apikey=46MMSGHA8VICBAWQ&outputsize=full"


import urllib.request

with urllib.request.urlopen(link) as url:
    data = json.loads(url.read().decode())

print(data)
type(data) # dict
data['Time Series (Daily)']

import pandas as pd
df = pd.DataFrame.from_dict(data['Time Series (Daily)'])
df = df.T



