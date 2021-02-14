import datetime

import numpy as np
import pandas as pd


import holidays

import matplotlib.pyplot as plt
from pandas.plotting._core import plot_frame

#%% 

"""
# Series 
"""

# Create 
pd.Series([1, 2, 3, np.nan, 6])
pd.Series([4, 3, 2], 
          index=['2015', '2016', '2017'], 
          name='Sales')
pd.Series([1, 2, 3], index=list('abc'))
pd.Series({'a': 1, 'b': 2})
pd.Series({'a': 1, 'b': 2}, index=['b', 'c'])

# Datum als index 
pd.Series([1, 2, 3, 4], name='Sales', index=pd.date_range('20130101', periods=4) )

# Verbinden
pd.concat([pd.Series([1, None]), pd.Series([10, 20])])
pd.Series([1, None]).append(pd.Series([10, 20])) # Achtung: not in place!

# Query
s = pd.Series([1, 2, 3, 4], name='Sales', index=list('abcd'))
s.loc['b']
s.iloc[1]
s[1] # wählt loc oder iloc aus, wenn eindeutig


# Operationen: +, -, /, *, ~
pd.Series([1, 2, 3]) + 1
pd.Series([1, 2, 3]) + pd.Series([3, 2, 1])
pd.Series([1, 2, 3]) * 2
pd.Series([1, 2, 3]) * pd.Series([1, 2, 3])
pd.Series([1, 2, 3])**3
~pd.Series([True, False]) # negation

# Ausgewählte Methoden
pd.Series([True, False]).all()
pd.Series([True, False]).any()
pd.Series([2, 3]).apply(lambda x: x**x)
pd.Series([2, 3]).astype(float)
pd.Series([1, 2, 3, 4, 5]).clip(2, 4)
pd.Series([2]).copy()
pd.Series([0.0, 1.0, np.nan]).count()
pd.Series([0, 1]).dot(pd.Series([1, 0]))
pd.Series([1, np.nan]).dropna()
pd.Series([1, 2, 3]).isin([2, 3])
pd.Series([1, np.nan]).isna()
pd.Series([1, np.nan]).notna()
pd.Series([1, np.nan]).map('-> {}'.format, na_action='ignore')



#%%

"""
# DataFrame 
"""

# Create
pd.DataFrame(np.random.randn(6, 4), 
             index=pd.date_range('20130101', periods=6), 
             columns=list('ABCD'))

pd.DataFrame([pd.Series([30, 35, 40], index=['a', 'b', 'c']), 
              pd.Series(['x', 'y', 'z'], index=['a', 'b', 'c'])],
              index=['X', 'Y']) # index bezieht sich auf Zeilen des DFs

pd.DataFrame([pd.Series([30, 35, 40], index=['a', 'b', 'c']), 
              pd.Series(['x', 'y', 'z'], index=['a', 'b', 'c'])],
              index=['X', 'Y']).T # = .transpose()

pd.DataFrame({'X': pd.Series([30, 35, 40]), 'Y': pd.Series(['x', 'y', 'z'])},
              index=[0,1,2])

pd.DataFrame({'col 1':[1,2,3],
              'col 2':[4,5,6]},
             index=['row 1', 'row 2', 'row 3']) 

pd.DataFrame([(1,'b', 3), 
              (4,'e', 5)],
             columns = ['a', 'b', 'c']) 

df = pd.DataFrame({'col 1':[1,2,3],'col 2':[4,5,6]},index=['row 1', 'row 2', 'row 3']) 
df2 = df.iloc[0:0].copy()
df2

#%%

"""
# Slicing
"""


df = pd.DataFrame({'col 1':[1,2,3,4],
                   'col 2':[4,5,6,7]},
                   index=['row 1', 'row 2', 'row 3', 'row 4']) 

## Slice by row and column label
df.loc['row 1']
df.loc['row 1', 'col 1']
df.loc[:, 'col 1']
df.loc[:, ['col 1', 'col 2']]
df.loc[['row 1', 'row 3']]
df.loc['row 2':'row 3', 'col 1':'col 2']

## Slice with []
df
df['col 1'] # Spalte
df[['col 1', 'col 2']] # 2 Spaltenimport numpy as np
df['col 1']['row 2'] # erste Spalte,  dann eine Zelle 
df['col 1'].notnull() 

## Slice by position (iloc)
df.iloc[[2]] # 3. Zeile 
df.iloc[2] # auch 3. Zeile, aber transponiert (völlig behindert)
df.iloc[:,:] # alles
df.iloc[:,1] # 2. Spalte

## Single cell with at
df.at['row 1', 'col 2'] # Zelle
df.at[1, 'n'] += 10 # also good to modify


#%%


"""
# Delete
"""

df.drop('row 1', inplace=True)
df
del df['col 1']
df
df.drop('col 2', axis=1) 
df
df = df[df['col 1']!=1]
df

cols = [c for c in df.columns if c.lower()[:4] != 'test']
df=df[cols]


# Load
df = pd.read_csv('data/olympics.csv', index_col=0, skiprows=1)
for col in df.columns:
    if col[:2]=='01':
        df.rename(columns={col:'Gold' + col[4:]}, inplace=True)
    if col[:2]=='02':
        df.rename(columns={col:'Silver' + col[4:]}, inplace=True)
    if col[:2]=='03':
        df.rename(columns={col:'Bronze' + col[4:]}, inplace=True)
    if col[:1]=='№':
        df.rename(columns={col:'#' + col[1:]}, inplace=True) 

df.head()



# query, select, conditions, boolean masks
df.dropna()[df['Gold']>0].sort_values('Gold', ascending=False)
len(df[(df['Gold'] > 0) | (df['Gold.1'] > 0)])
df[(df['Gold.1'] > 0) & (df['Gold'] == 0)]
df['Gold'].isin([0, 4])
df[df['Gold'].isin([5, 6])]
df['Country'] = df.index
df[df['Country'].str.startswith('A')]


b = df['A'].notnull() & (df['A'] > 0)
b
df.loc[b]

## With loc
df = pd.DataFrame({'col 1':[1,2,3,4],
                   'col 2':[4,5,6,7]},
                   index=['row 1', 'row 2', 'row 3', 'row 4']) 
df.loc[:, 'col 1'] > 1 # boolean mask rows
df.loc[df.loc[:, 'col 1'] > 2, :] # select with mask
df.loc[df.loc[:, 'col 1'] > 2, 'col 2'] = 99 # assign with mask
df.loc[df['col 1'].isin([1, 2])]


"""
# Indexing
"""

df['Country'] = df.index # index to column
df.set_index('Gold') # column to index

df.reset_index()

purchase_1 = pd.Series({'Name': 'Chris',
                        'Item Purchased': 'Dog Food',
                        'Cost': 22.50})
purchase_2 = pd.Series({'Name': 'Kevyn',
                        'Item Purchased': 'Kitty Litter',
                        'Cost': 2.50})
purchase_3 = pd.Series({'Name': 'Vinod',
                        'Item Purchased': 'Bird Seed',
                        'Cost': 5.00})
df = pd.DataFrame([purchase_1, purchase_2, purchase_3], 
                  index=['Store 1', 'Store 1', 'Store 2'])

df = df.set_index([df.index, 'Name']) # extend index
df.index.names = ['Location', 'Name'] # name index

"""
# Indexing
"""

df = pd.DataFrame({'month': [1, 2, 3, 4],
                   'year': [10, 20, 30, 40],
                   'sale': [100, 200, 300, 400]})
df.set_index(pd.Index([3, 4, 5, 6]))
df
df.set_index(pd.Index(range(df.shape[0])))
df

# Re-indexing
s = pd.Series(np.random.randn(5), index=['a', 'b', 'c', 'd', 'e'])
s
s.reindex(['e', 'b', 'f', 'd']) # => Reihenfolge geändert, und f gab es nicht, daher nun NaN



"""
# Append, concat, ...
"""

# append row
df = df.append(pd.Series(data={'Cost': 3.00, 'Item Purchased': 'Kitty Food'}, 
                         name=('Store 2', 'Kevyn')))
# append rows
df.append(df.copy())


# Missing values




# Ausgewählte Methoden
df
df.dtypes
df.size # Anzahl Elemente = nRows * nCols
df.shape # (nRows, nCols), Tuple
df.shape[0]

df.head(3)
df.tail(2)
df.describe()

df.T # transponieren

for index, row in df.iterrows(): # Iterate rows
   print(row['month'])




#%%
## Verändern

# modifiziere spaltenweise (hier nur eine)
df = pd.DataFrame({'A': [1, 2, 3, 4],
                   'B': [10, 20, 30, 40],
                   'C': [100, 200, 300, 400]})
    
    

df['A'] = 3 # ganze Spalte!
df
df['A'] = range(2,6)  # mit iterable



# Spaltennamen ändern
df.columns = ['X','Y','Z']
df



df['a'] = df['a'].apply(np.square)
df['a'] = df['a'].apply(lambda x: x+1)
df
df['ab'] = df['a'] + df['b']


# modifiziere zeilenweise
df.apply(lambda row: row['a']+row['b'], axis=1)
def my(row):
    return row['a']+row['b']
df.apply(my, axis=1)


# transform
df = pd.DataFrame({'A': range(3), 'B': range(1, 4)})
df.transform(lambda x: x + 1)
df.B = df.B.transform(lambda x: x + 7)
df

## Erweitern

df =  pd.DataFrame({'A': range(3), 'B': range(1, 4)})
df = df.append(pd.DataFrame({'A':[9],'B':[99]})) # Zeile
df = df.assign(E=df['A'] * 2) # Spalte
df['F'] = -1 * df['E'] # Spalte
df = df.assign(G=lambda x: x.A < 2) # oder so
df



df = pd.DataFrame({'A': [1, 2, 3, 4],'B': [4, 3, 2, 1]})
df
df['C']=df.min(axis=1)
df




"""
# Exportieren
"""
df.to_records(index=False).tolist()


###
# Beispiel mit Feiertagen und zusätzlichen 'Features'
###


mo = datetime.datetime(2019, 8, 12) # mo
mo.weekday()
so = datetime.datetime(2019, 8, 11) # so
so.weekday()

holis = holidays.CountryHoliday('DE')
holis = holidays.Germany()
holisBayern = holidays.Germany(prov='BY')
datetime.datetime(2019, 10, 3) in holis
datetime.datetime(2019, 8, 15) in holis # Mariä Himmelfahrt in Bayern
datetime.datetime(2019, 8, 15) in holisBayern 
datetime.datetime(2019, 10, 3) in holisBayern # der 2. Okt  is auch in Bayern einer

m = 8
dates = pd.date_range('20190814', periods=m)
df = pd.DataFrame({'date':dates, 'b':list(range(1, m+1))})
df

dates = pd.date_range(start='20190101', end='2019-12-31 23:59', freq='H')



# neue Spalten mit Wochentagindikatoren
data = data.assign(mon=data.apply(lambda row: 1 if row['date'].weekday()==0 else 0, axis=1))
data = data.assign(tue=data.apply(lambda row: 1 if row['date'].weekday()==1 else 0, axis=1))
data = data.assign(wed=data.apply(lambda row: 1 if row['date'].weekday()==2 else 0, axis=1))
data = data.assign(thu=data.apply(lambda row: 1 if row['date'].weekday()==3 else 0, axis=1))
data = data.assign(fri=data.apply(lambda row: 1 if row['date'].weekday()==4 else 0, axis=1))
data = data.assign(sat=data.apply(lambda row: 1 if row['date'].weekday()==5 else 0, axis=1))
data = data.assign(sun=data.apply(lambda row: 1 if row['date'].weekday()==6 else 0, axis=1))

# neue Spalte mit Feiertagindikator
data = data.assign(holiday=data.apply(lambda row: 1 if row['date'] in holisBayern else 0, axis=1))
data


plt.plot(data['date'], data['b'])
plt.show()


#%%

"""
# Aggregieren, aggregate, grouping
"""

data = pd.DataFrame({'a': [1,2,3,1,2,3,1],
                     'b':[1,1,1,2,2,3,1],
                     'c':[2,2,5,5,1,1,1]})

data.aggregate(np.sum) # 1 aggregate per column 
data.aggregate(np.sum, axis="columns") # 1 aggregate per row

data.groupby('a').sum()

data.groupby('a').sum()['b'].to_numpy()

df = data.groupby(['a','b']).sum() # multiple columns
print(df)
df.reset_index() # index back as column

# unique combinations of some columns
grp = df.groupby('a', axis=0).size()



#%% 
# sqldf - nur selects

import pandasql as ps
ps.sqldf("select * from df where a>1", locals())


#%% 
# Beispiel: plotte Zeitreihendatenpunkte, die verfügbar sind.

s = pd.to_datetime(
        pd.Series(['2014-05-01 18:47:05.069722', '2014-05-01 18:47:05.119994', 
                   '2014-05-02 17:47:05.178768', '2014-05-07 18:47:05.230071', 
                   '2014-05-02 18:47:05.230071', '2014-05-12 01:47:05.280592', 
                   '2014-05-02 19:47:05.332662', '2014-05-13 18:47:05.385109', 
                   '2014-05-04 18:47:05.436523'], name='dt'))

df = pd.DataFrame({'dates': s, 
                   'ts': [ datetime.timestamp(d) for d in s.tolist() ]})  
df
df.plot(kind='line', style='.', x='dates', y='ts',  color='red')
plt.show()



# Datums-Range mit Tagen
start_date = s.min().date()
end_date = s.max().date() + datetime.timedelta(days=1)
dates = [start_date + datetime.timedelta(n) for n in range(int ((end_date - start_date).days))]
df = pd.DataFrame({'dates': dates, 'ind':range(len(dates))})  
df




#%%

"""
# Strings
"""


df.Zip.str[:3]  # indexing with str, to get substring


#%%
# Meine Funktionen

def toTimestamp(a: pd._libs.tslibs.timestamps.Timestamp):
    return (a.asm8 - np.datetime64('1970-01-01T00:00:00')) / np.timedelta64(1, 's')





"""
# Special transformations
"""

# One-hot encoding
df = pd.DataFrame({'a': [1,2,2,3,3,3]})
df = pd.get_dummies(df.a, prefix='a')

# Transpose, melt
df = pd.DataFrame({'number of chickens': [4,5,6],
                   'number of eggs': [11,15,22]},
                  index=['farm 1', 'farm 2', 'farm 3']) 
print(df.head())
df.reset_index().melt(id_vars=['index'], 
                      var_name='measure', 
                      value_name='value')


