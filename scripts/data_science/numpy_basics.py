import numpy as np

# ARRAYS
my_list = [1,2,3]
my_list

np.array(my_list)

my_matrix = [[1,2,3],[4,5,6],[7,8,9]]
my_matrix

np.array(my_matrix)

# BUILT-IN METHODS
np.arange(0,10) # Return evenly spaced values within a given interval.
np.arange(0,11,2)

np.zeros(3) # Generate arrays of zeros or ones
np.zeros((5,5))

np.ones(3)
np.ones((3,3))

np.linspace(0,10,3) # Return evenly spaced numbers over a specified interval.
np.linspace(0,10,50)

np.eye(4) # Creates an identity matrix

# RANDOM
np.random.rand(2) # Create an array of the given shape and populate it with random samples from a uniform distribution over [0, 1).
np.random.rand(5,5)

np.random.randn(2) # Return a sample (or samples) from the "standard normal" distribution. Unlike rand which is uniform
np.random.randn(5,5)

np.random.randint(1,100) # Return random integers from low (inclusive) to high (exclusive).
np.random.randint(1,100,10)

# ARRAY ATTRIBUTES AND METHODS
arr = np.arange(25)
ranarr = np.random.randint(0,50,10)

print(arr)
print(ranarr)

# RESHAPE
arr.reshape(5,5) # Returns an array containing the same data with a new shape.

ranarr

# These are useful methods for finding max or min values. Or to find their index locations using argmin or argmax
ranarr.max()
ranarr.argmax()
ranarr.min()
ranarr.argmin()

# SHAPE
# Vector
arr.shape

# Notice the two sets of brackets
arr.reshape(1,25)

arr.reshape(1,25).shape

arr.reshape(25,1)

arr.reshape(25,1).shape

# DTYPE
arr.dtype # You can also grab the data type of the object in the array

# OPERATIONS

arr = np.arange(0,10)
arr + arr
arr * arr
arr - arr
arr**3

# UNIVERSAL ARRAY FUNCTIONS

#Taking Square Roots
np.sqrt(arr)

#Calcualting exponential (e^)
np.exp(arr)

np.max(arr) #same as arr.max()

np.sin(arr)

np.log(arr)
