from builtins import type
from datetime import datetime
from datetime import timedelta

import pandas as pd

# %%

dt = datetime(2019, 8, 12)
dt = datetime(2019, 8, 12, 12, 5)
print(dt)
print(type(dt))
print(dt.date())
print(dt.time())
print('{}:{}'.format(dt.hour, dt.minute))
print(dt.timetuple().tm_yday)  # day of year

# %%

# datetime from string
date_str = '2015-01-01 09:15:00'
print(datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S'))

# %%

# Add, subtract
d = datetime.today() - timedelta(days=1)  # yesterday
print(d)

# %%
"""
# Pandas
"""

dates = pd.date_range('20130101', periods=6, freq='s')
print(dates)
pd.to_datetime(dates, unit='D')  # not working
dates = dates.date
dates

dt = pd.to_datetime('2019-08-12 12:05:00', format='%Y-%m-%d %H:%M:%S')
type(dt)
dt.date()
dt.date().year
dt.year
dt.hour

# als Serie
s = pd.Series(['2019-08-12 12:05:00', '2019-08-13 13:07:00'])
s
arr = s.values
arr
arr.dtype

# to datetime
s2 = pd.to_datetime(s, format='%Y-%m-%d %H:%M:%S')
s2
pd.to_datetime(s2.values).date
pd.to_datetime(s2.values).year
pd.to_datetime(s2.values, unit='Y').date

pd.to_datetime(['2019-08-12 12:05:00', '2019-08-12 12:05:00'], format='%Y-%m-%d %H:%M:%S').year
