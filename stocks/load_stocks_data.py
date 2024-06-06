import json
import urllib.request

link = "https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=ADS.DE&apikey=46MMSGHA8VICBAWQ&outputsize=full"

with urllib.request.urlopen(link) as url:
    data = json.loads(url.read().decode())

print(data)
type(data)  # dict
data['Time Series (Daily)']

df = pd.DataFrame.from_dict(data['Time Series (Daily)'])
df = df.T
