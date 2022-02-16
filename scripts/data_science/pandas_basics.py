import numpy as np
import pandas as pd

# SERIES
labels = ['a','b','c']
my_list = [10,20,30]
arr = np.array([10,20,30])
d = {'a':10,'b':20,'c':30}

pd.Series(data=my_list)
pd.Series(data=my_list,index=labels)
pd.Series(my_list,labels)
pd.Series(arr)
pd.Series(arr,labels)
pd.Series(d)

# DATA in a SERIES

pd.Series(data=labels) # A pandas Series can hold a variety of object types

# Even functions (although unlikely that you will use this)
pd.Series([sum,print,len])

# USING AN INDEX
ser1 = pd.Series([1,2,3,4],index = ['USA', 'Germany','USSR', 'Japan'])

ser2 = pd.Series([1,2,5,4],index = ['USA', 'Germany','Italy', 'Japan'])

ser1['USA']
ser1 + ser2

# MISSING DATA

df = pd.DataFrame({'A':[1,2,np.nan],
                  'B':[5,np.nan,np.nan],
                  'C':[1,2,3]})
df.dropna()
df.dropna(axis=1)
df.dropna(thresh=2)
df.fillna(value='FILL VALUE')
df['A'].fillna(value=df['A'].mean())

# GROUPBY

# Create dataframe
data = {'Company':['GOOG','GOOG','MSFT','MSFT','FB','FB'],
       'Person':['Sam','Charlie','Amy','Vanessa','Carl','Sarah'],
       'Sales':[200,120,340,124,243,350]}

df = pd.DataFrame(data)
df.groupby('Company') # .groupby() method to group rows together based off of a column name. For instance let's group based off of Company. This will create a DataFrameGroupBy object
by_comp = df.groupby("Company")
by_comp.mean()
df.groupby('Company').mean()
by_comp.std()
by_comp.min()
by_comp.max()
by_comp.count()
by_comp.describe()
by_comp.describe().transpose()
by_comp.describe().transpose()['GOOG']

# OPERATIONS

df = pd.DataFrame({'col1':[1,2,3,4],'col2':[444,555,666,444],'col3':['abc','def','ghi','xyz']})
df.head()

df['col2'].unique()
df['col2'].nunique()
df['col2'].value_counts()

#Select from DataFrame using criteria from multiple columns
newdf = df[(df['col1']>2) & (df['col2']==444)]

def times2(x):
    return x*2
df['col1'].apply(times2)
df['col3'].apply(len)
df['col1'].sum()

del df['col1'] # Permanently Removing a Column

df.columns
df.index

df.sort_values(by='col2') #inplace=False by default
df.isnull()

# Drop rows with NaN Values
df.dropna()

df = pd.DataFrame({'col1':[1,2,3,np.nan],
                   'col2':[np.nan,555,666,444],
                   'col3':['abc','def','ghi','xyz']})
df.head()
df.fillna('FILL') # Filling in NaN values with something else

data = {'A':['foo','foo','foo','bar','bar','bar'],
     'B':['one','one','two','two','one','one'],
       'C':['x','y','x','y','x','y'],
       'D':[1,3,2,5,4,1]}

df = pd.DataFrame(data)
df.pivot_table(values='D',index=['A', 'B'],columns=['C'])
