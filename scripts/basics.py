
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
# Loops
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
# Open file
with open(file_name, 'w') as f: # 'w', 'r', 'wb', 'a'
    f.readline()
f.close()


# Save object to file
import pickle
with open(file_name, 'wb') as f:
    pickle.dump(text, f, pickle.HIGHEST_PROTOCOL)

##################
# Doesn't use regular codex
import codecs
with codecs.open(file_name, 'w', encoding='utf-8') as f:
    f.write(text)


import time

start_time = time.time()
print("My program took", time.time() - start_time, "to run")





# Sources:
## https://treyhunner.com/2016/04/how-to-loop-with-indexes-in-python/
## https://www.w3schools.com/python
