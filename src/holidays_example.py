from datetime import datetime

import holidays
import matplotlib.pyplot as plt
import pandas as pd

# %%
"""
# Beispiel mit Feiertagen und zusätzlichen 'Features'
"""

mo = datetime(2019, 8, 12)  # mo
mo.weekday()
so = datetime(2019, 8, 11)  # so
so.weekday()

holis = holidays.CountryHoliday('DE')
holis = holidays.Germany()
holisBayern = holidays.Germany(prov='BY')
datetime(2019, 10, 3) in holis
datetime(2019, 8, 15) in holis  # Mariä Himmelfahrt in Bayern
datetime(2019, 8, 15) in holisBayern
datetime(2019, 10, 3) in holisBayern  # der 2. Okt  is auch in Bayern einer

m = 8
dates = pd.date_range('20190814', periods=m)
df = pd.DataFrame({'date': dates, 'b': list(range(1, m + 1))})
df

dates = pd.date_range(start='20190101', end='2019-12-31 23:59', freq='H')

# neue Spalten mit Wochentagindikatoren
data = data.assign(mon=data.apply(lambda row: 1 if row['date'].weekday() == 0 else 0, axis=1))
data = data.assign(tue=data.apply(lambda row: 1 if row['date'].weekday() == 1 else 0, axis=1))
data = data.assign(wed=data.apply(lambda row: 1 if row['date'].weekday() == 2 else 0, axis=1))
data = data.assign(thu=data.apply(lambda row: 1 if row['date'].weekday() == 3 else 0, axis=1))
data = data.assign(fri=data.apply(lambda row: 1 if row['date'].weekday() == 4 else 0, axis=1))
data = data.assign(sat=data.apply(lambda row: 1 if row['date'].weekday() == 5 else 0, axis=1))
data = data.assign(sun=data.apply(lambda row: 1 if row['date'].weekday() == 6 else 0, axis=1))

# neue Spalte mit Feiertagindikator
data = data.assign(holiday=data.apply(lambda row: 1 if row['date'] in holisBayern else 0, axis=1))
data

plt.plot(data['date'], data['b'])
plt.show()
