# CONTENTS
## COLLECTIONS
## # COUNTER
## # DEFAULTDICT
## # NAMEDTUPLE
## MATH

## TIME
## DATETIME
## TIMEIT

##################
# COLLECTIONS MODULE
# COUNTER
from collections import Counter

s = 'How many times does each word show up in this sentence word times each each word'

words = s.split()

Counter(words)
# Methods with Counter()
c = Counter(words)

c.most_common(2)
sum(c.values())                 # total of all counts
c.clear()                       # reset all counts
list(c)                         # list unique elements
set(c)                          # convert to a set
dict(c)                         # convert to a regular dictionary
c.items()                       # convert to a list of (elem, cnt) pairs
Counter(dict(list_of_pairs))    # convert from a list of (elem, cnt) pairs
c.most_common()[:-n-1:-1]       # n least common elements
c += Counter()                  # remove zero and negative counts


# DEFAULTDICT
from collections import defaultdict

d  = defaultdict(object) # can't raise KeyError
d['one']

d = defaultdict(lambda: 0) #change initialization
d['one']


# NAMEDTUPLE
from collections import namedtuple

Dog = namedtuple('Dog',['age','breed','name'])

sam = Dog(age=2,breed='Lab',name='Sammy')
frank = Dog(age=2,breed='Shepard',name='Frankie')

##################
# MATH
import math

# ROUNDING
value = 4.35
math.floor(value)
math.ceil(value)
round(value)

# CONSTANTS
math.pi

from math import pi
pi

math.e
math.tau
math.inf
math.nan

# LOGARITHMIC VALUES
math.e
math.log(math.e)
math.log(10)
math.e ** 2.302585092994046

# math.log(x,base)
math.log(100,10)

# Radians
math.sin(10)
math.degrees(pi/2)
math.radians(180)

# RANDOM
import random

random.randint(0,100)

# Seed will give you the same rand values every time
# The value 101 is completely arbitrary, you can pass in any number you want
random.seed(101)
# You can run this cell as many times as you want, it will always return the same number
random.randint(0,100)

# Random number from list
mylist = list(range(0,20))
random.choice(mylist)

random.choices(population=mylist,k=10) # Take a sample size, allowing picking elements more than once.
random.sample(population=mylist,k=10) # Once an item has been randomly picked, it can't be picked again.

random.shuffle(mylist)

# Continuous, random picks a value between a and b, each value has equal change of being picked.
random.uniform(a=0,b=100)

# Normal/Gaussian distribution
random.gauss(mu=0,sigma=1)

##################
# TIME
import time

start_time = time.time()
print("My program took", time.time() - start_time, "to run")

# STEP 1: Get start time
start_time = time.time()
# Step 2: Run your code you want to time
result = func_one(1000000)
# Step 3: Calculate total time elapsed
end_time = time.time() - start_time
end_time

## DATETIME
import datetime
t = datetime.time(4, 20, 1)
print(t)
print('hour  :', t.hour)
print('minute:', t.minute)
print('second:', t.second)
print('microsecond:', t.microsecond)
print('tzinfo:', t.tzinfo)

print('Earliest  :', datetime.time.min)
print('Latest    :', datetime.time.max)
print('Resolution:', datetime.time.resolution)


today = datetime.date.today()
print(today)
print('ctime:', today.ctime())
print('tuple:', today.timetuple())
print('ordinal:', today.toordinal())
print('Year :', today.year)
print('Month:', today.month)
print('Day  :', today.day)

print('Earliest  :', datetime.date.min)
print('Latest    :', datetime.date.max)
print('Resolution:', datetime.date.resolution)


##################
# TIMEIT
import timeit

setup = '''
def func_one(n):
    return [str(num) for num in range(n)]
'''

stmt = 'func_one(100)'
timeit.timeit(stmt,setup,number=100000)

setup2 = '''
def func_two(n):
    return list(map(str,range(n)))
'''

stmt2 = 'func_two(100)'
timeit.timeit(stmt2,setup2,number=100000)

timeit.timeit(stmt,setup,number=1000000)
timeit.timeit(stmt2,setup2,number=1000000)
# https://docs.python.org/3/library/timeit.html
