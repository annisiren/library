import numpy as np

#Creating sample array
arr = np.arange(0,11)

#Show
arr

# BRACKET INDEXING AND SELECTION
#Get a value at an index
arr[8]

#Get values in a range
arr[1:5]

#Get values in a range
arr[0:5]

# BROADCASTING

#Setting a value with index range (Broadcasting)
arr[0:5]=100

#Show
arr

# Reset array, we'll see why I had to reset in  a moment
arr = np.arange(0,11)

#Show
arr

#Important notes on Slices
slice_of_arr = arr[0:6]

#Show slice
slice_of_arr

#Change Slice
slice_of_arr[:]=99

#Show Slice again
slice_of_arr

arr

#To get a copy, need to be explicit
arr_copy = arr.copy()

arr_copy

# INDEXING a 2D ARRAY

arr_2d = np.array(([5,10,15],[20,25,30],[35,40,45]))

#Show
arr_2d

#Indexing row
arr_2d[1]

# Format is arr_2d[row][col] or arr_2d[row,col]

# Getting individual element value
arr_2d[1][0]

# Getting individual element value
arr_2d[1,0]

# 2D array slicing

#Shape (2,2) from top right corner
arr_2d[:2,1:]

#Shape bottom row
arr_2d[2]

#Shape bottom row
arr_2d[2,:]

# FANCY INDEXING

#Set up matrix
arr2d = np.zeros((10,10))

#Length of array
arr_length = arr2d.shape[1]

#Set up array

for i in range(arr_length):
    arr2d[i] = i

arr2d

arr2d[[2,4,6,8]]

#Allows in any order
arr2d[[6,4,2,7]]

# SELECTION

arr = np.arange(1,11)
arr

arr > 4

bool_arr = arr>4

bool_arr

arr[bool_arr]

arr[arr>2]

x = 2
arr[arr>x]
