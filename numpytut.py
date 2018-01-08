import numpy as np 
from matplotlib import pyplot as plt   

# Numpy array is of fixed size
arr = np.array([1,2,34])
# Two d arrays in numpy
# To create a numpy array of zeroes
a = np.zeros((4,4))
# Make all the element in the first column as 0
a[:,0] = 2
print(a)
# Make all the elements in the second row as 0
a[1, :] = 3
print(a)

# Unique and argmax functions in numpy 
arr = np.asarray([1,2,3,4,56,73,4,4,4,4,4,43])
b = np.unique(arr, return_counts=True)
# Gives each element along with its count next to it
# print(b[1].argmax)
print(b)