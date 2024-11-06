import json
import urllib.request

import pandas as pd

API_KEY = "46MMSGHA8VICBAWQ"
link = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=ADS.DE&apikey={API_KEY}&outputsize=full"

with urllib.request.urlopen(link) as url:
    data = json.loads(url.read().decode())

df = pd.DataFrame.from_dict(data['Time Series (Daily)'])
df = df.T

print(df)
