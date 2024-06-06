#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JSON examples.
"""

import json

import pandas as pd

df1 = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
df2 = pd.DataFrame({'c': [9, 8], 'd': [7, 6]})
dct1 = df1.to_dict()  # any orientation
dct2 = df2.to_dict()
response = {"message": "success", "payload": {'table1': dct1, 'table2': dct2}}
ret = json.dumps(response, indent=4)
print(ret)
