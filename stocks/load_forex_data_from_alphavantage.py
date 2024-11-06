import json
from urllib import request

API_KEY = "46MMSGHA8VICBAWQ"
url = (f'https://www.alphavantage.co/query?function=FX_INTRADAY&'
       f'from_symbol=EUR&'
       f'to_symbol=USD&'
       f'interval=1min'
       f'&apikey={API_KEY}')

with request.urlopen(url) as url:
    data = json.loads(url.read().decode())

print(data)
