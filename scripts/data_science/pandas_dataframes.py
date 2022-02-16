import pandas as pd
import numpy as np

from numpy.random import randn
np.random.seed(101)

df = pd.DataFrame(randn(5,4),index='A B C D E'.split(),columns='W X Y Z'.split())

# SELECTION AND INDEXING

df['W']

# Pass a list of column names
df[['W','Z']]

# SQL Syntax (NOT RECOMMENDED!)
df.W

type(df['W']) # DataFrame Columns are just Series

df['new'] = df['W'] + df['Y'] # Creating a new column:

df.drop('new',axis=1) # Removing Columns
# Not inplace unless specified!
df.drop('new',axis=1,inplace=True)
df.drop('E',axis=0)

# Selecting Rows
df.loc['A']
df.iloc[2] # Select based off of position instead of label
df.loc['B','Y'] # Selecting subset of rows and columns
df.loc[['A','B'],['W','Y']]

# CONDITIONAL SELECTION

df>0
df[df>0]
df[df['W']>0]
df[df['W']>0]['Y']
df[df['W']>0][['Y','X']]
df[(df['W']>0) & (df['Y'] > 1)]

# INDEX DETAILS

# Reset to default 0,1...n index
df.reset_index()

newind = 'CA NY WY OR CO'.split()
df['States'] = newind

df.set_index('States')
df.set_index('States',inplace=True)

# MULTI-INDEX and INDEX HIERARCHY

# Index Levels
outside = ['G1','G1','G1','G2','G2','G2']
inside = [1,2,3,1,2,3]
hier_index = list(zip(outside,inside))
hier_index = pd.MultiIndex.from_tuples(hier_index)

df = pd.DataFrame(np.random.randn(6,2),index=hier_index,columns=['A','B'])

df.loc['G1']
df.loc['G1'].loc[1]
df.index.names
df.index.names = ['Group','Num']
df.xs('G1')
df.xs(['G1',1])
df.xs(1,level='Num')
