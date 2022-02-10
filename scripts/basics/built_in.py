##################
# BUILT IN FUNCTIONS
## all() & any()
lst = [True,True,False,True]
all(lst) #False
any(lst) # True

## complex() - complex numbers
# Create 2+3j
complex(2,3)

# NUMBERS
hex(246)

bin(1234)

pow(3,4)

abs(-3.14)

round(3,2)



##################
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
