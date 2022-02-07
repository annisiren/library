
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

##################
# MISC
## time
import time

start_time = time.time()
print("My program took", time.time() - start_time, "to run")





# Sources:
## https://treyhunner.com/2016/04/how-to-loop-with-indexes-in-python/
## https://www.w3schools.com/python
