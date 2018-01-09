# Acquire all the core libraries
import numpy as np 
import pandas as pd 
from matplotlib import pyplot as plt 

ds = pd.read_csv('./train.csv')
print(ds.shape)

data = ds.values 
print(data.shape)
# mnist is the dataset of 42000 rows and 785 columns in
# which first column comprises of all the labels
# and all remaining 784 columns comprises of training data

# For the labels I want all of the rows and the 0th column
# For the training data I want all of the rows and the 1st column
y_train = data[:,0]
X_train = data[:,1]
# Print the shape of these
print(y_train.shape)
print(X_train.shape)    

# Setup the id of the first figure
plt.figure(0)
idx = 104
print(y_train[idx])     #Print the label at the random index 1204

# X_train is of the form 784*1 so we want to reshape it to 28*28 and map it into grayscale
plt.imshow(X_train[idx].reshape((28,28)),cmap='gray')
plt.show()


def distance(x1,x2):
    # Here compute the euclidean distance between the two points representing the two vectors
    return np.sqrt(((x1-x2)**2).sum())

# Basically k is the nearest neighbours count which is generally taken as odd
# k should not be too high or too low
# Here small x is the query point
def knn(X_train,x,y_train,k=5):
    vals = []
    # Iterate over the training data and find the distance of all the points in the dataset from the query point x
    for ix in range(X_train.shape[0]):
        v = [distance(x,X_train[ix,:]),y_train[ix]]
        vals.append(v)
    # Sort all the values that we got
    updated_vals = sorted(vals,key=lambda x:x[0])
    # Pick the top k values only
    pred_arr = np.asarray(updated_vals[:k])
    pred_arr = np.unique(pred_arr[:,1],return_counts = True)
    pred = pred_arr[1].argmax()
    return pred_arr, pred_arr[0][pred]    

idq = int(np.random.random() * X_train.shape[0])
q = X_train[idq]

res = knn(X_train[:10000],q,y_train[:10000],k=7)
print(res)
print(y_train[idq])

plt.figure(0)
plt.imshow(q.reshape((28,28)),cmap='gray')
plt.show()
