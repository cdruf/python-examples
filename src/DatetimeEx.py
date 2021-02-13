import datetime
from builtins import type

# new date
dt = datetime.datetime(2019, 8, 12)
dt
type(dt)
dt.date()

# new datetime
datetime.datetime(2019, 8, 12, 12, 5)

# datetime from string
dateStr = '2015-01-01 09:15:00' 
datetime.datetime.strptime(dateStr, '%Y-%m-%d %H:%M:%S')


# Attribute
d = datetime.datetime(2019, 10, 3, 12, 5)
d.hour
d.minute
d.date()

d.timetuple().tm_yday # day of year

dates = pd.date_range('20130101', periods=6, freq='s')
dates
pd.to_datetime(dates,unit='D') # not working
dates = dates.dt.date


# 
d = datetime.datetime.today() - datetime.timedelta(days=1) # gestern



###
# Pandas
# Das is wirklich unfassbar behindert!!!!!
###

# einzeln
import pandas as pd
dt = pd.to_datetime('2019-08-12 12:05:00', format='%Y-%m-%d %H:%M:%S')
type(dt)
dt.date()
dt.date().year
dt.year
dt.hour

# als Serie
s = pd.Series(['2019-08-12 12:05:00', '2019-08-13 13:07:00'])
s
type(s)
arr = s.values
arr
type(arr)
arr.shape
arr.dtype
np.char.split("sadf asdfsad")
 


# nach datetime 
s2 = pd.to_datetime(s, format='%Y-%m-%d %H:%M:%S')
s2
pd.to_datetime(s2.values).date
pd.to_datetime(s2.values).year
pd.to_datetime(s2.values,unit='Y').date

pd.to_datetime(['2019-08-12 12:05:00', '2019-08-12 12:05:00'], format='%Y-%m-%d %H:%M:%S').year
pd.to_datetime(data['ride_departure'], format='%Y-%m-%d %H:%M:%S').year


