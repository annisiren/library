
##################
# STRING MANIPULATION
str = ['Hello', 'World']
str1 = " "
str1.join(str)

str = ['Hello', 'World']
' '.join(str)

# Remove all whitespace

str.strip()


##################
# LOOPS
## FOR
fruits = ["apple", "banana", "cherry"]
for x in fruits:
      print(x)

for x in range(2, 30, 3): # start, finish, increment
      print(x)
else:
      print("Finally finished!")

### ENUMERATE - keep track of index
presidents = ["Washington", "Adams", "Jefferson", "Madison", "Monroe", "Adams", "Jackson"]
for num, name in enumerate(presidents, start=1):
    print("President {}: {}".format(num, name))

### ZIP -- loop over multiple lists at the same time
colors = ["red", "green", "blue", "purple"]
ratios = [0.2, 0.3, 0.1, 0.4]
for color, ratio in zip(colors, ratios):
    print("{}% {}".format(ratio * 100, color))


## WHILE
colors = ["red", "green", "blue", "purple"]
i = 0
while i < len(colors):
    print(colors[i])
    i += 1

##################
# FUNCTION
## *args //unknown amount of arguments
def myfunc(*args):
    return sum(args)*.05

myfunc(40,60,20)

## **kwargs //keyword arguments
def myfunc(**kwargs):
    if 'fruit' in kwargs:
        print(f"My favorite fruit is {kwargs['fruit']}")  # review String Formatting and f-strings if this syntax is unfamiliar
    else:
        print("I don't like fruit")

myfunc(fruit='pineapple')

##LAMBDA
#lambda var: expression, list
my_nums = [1,2,3,4,5]
list(map(lambda num: num ** 2, my_nums))
list(filter(lambda n: n % 2 == 0, my_nums))

##################
## MAP, REDUCE, FILTER, ZIP, ENUMERATE
## MAP

a = [1,2,3,4]
b = [5,6,7,8]
c = [9,10,11,12]

list(map(lambda x,y,z:x+y+z,a,b,c))

## REDUCE
from functools import reduce
lst =[47,11,42,13]
reduce(lambda x,y: x+y,lst)

## FILTER
lst =range(20)
list(filter(lambda x: x%2==0,lst))

## ZIP
x = [1,2,3]
y = [4,5,6]

# Zip the lists together
list(zip(x,y))

##ENUMERATE
months = ['March','April','May','June']

list(enumerate(months,start=3))


##################
# BUILT IN FUNCTIONS
## all() & any()
lst = [True,True,False,True]
all(lst) #False
any(lst) # True

## complex() - complex numbers
# Create 2+3j
complex(2,3)


##################
# DECORATOR
def new_decorator(func):

    def wrap_func():
        print("Code would be here, before executing the func")

        func()

        print("Code here will execute after the func()")

    return wrap_func

def func_needs_decorator():
    print("This function is in need of a Decorator")

@new_decorator
def func_needs_decorator():
    print("This function is in need of a Decorator")

##################
# GENERATOR
# Generator function for the cube of numbers (power of 3)
def gencubes(n):
    for num in range(n):
        yield num**3
        # no return
for x in gencubes(10):
    print(x)

# built-in functions
# next()
def simple_gen():
    for x in range(3):
        yield x
# Assign simple_gen
g = simple_gen()
print(next(g))

# iter()
s = 'hello'

#Iterate over string
for let in s:
    print(let)
s_iter = iter(s)
next(s_iter)

# Sources:
## https://treyhunner.com/2016/04/how-to-loop-with-indexes-in-python/
## https://www.w3schools.com/python
