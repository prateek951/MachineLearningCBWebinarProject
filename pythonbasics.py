# lists in python
a = [1,24,4,5,5,"Hello"]
print(a)

#Slicing in lists
b = a[0:3]
print(b)
print(a[2:])

# Dictionaries
prices = {
    'mango' : 100,
    'orange' : 200,
    'banana' : [10,20,30]
}
print(type(prices))
# Iterate over all the keys
print(prices.keys())
print(prices.values())

# Loops
i = 1;
while i<=10:
    print(i)
    i+=1

# Using the math package(packages are like boxes)
#  import math 
from math import sqrt as sq
 math.log10(100)
 math.sqrt(2)

#  Scientific Computation part 
import numpy as np

# Creating arrays in numpy