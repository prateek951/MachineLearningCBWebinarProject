import numpy as np 
from matplotlib import pyplot as plt 

# Take mean along the x and y directions
mean_01 = np.array([3,4])
# Take the covariance matrix
cov_01 = np.array([[1.0,-0.5],[-0.5,1.0]])
mean_02 = np.array([0.0,0.0])
cov_02 = np.array([[1.0,.5],[0.5,0.6]])

# Generate the random distribution
# Generating 200 samples for the apples
dist_01 = np.random.multivariate_normal(mean_01,cov_01,200)
# Generating 200 samples for the lemons as well
dist_02 = np.random.multivariate_normal(mean_02,cov_02,200)
print(dist_01.shape)
print(dist_02.shape)
plt.figure(0)
for x in range(dist_01.shape[0]):
    plt.scatter(dist_01[x,0],dist_01[x,1],color='red')
    plt.scatter(dist_02[x,0],dist_02[x,1],color='yellow')
plt.show()

# Training data preparation
# 400 samples - 200 apples,200 lemons

labels = np.zeros((400,1))
# For lemons till 200 we will set it to 0
# From 200 onwards we have apples so for apples set 1.0
labels[200:] = 1.0

X_data = np.zeros((400,2))
# In first 200 rows put the data for apples
X_data[:200,:] = dist_01
# In next 200 rows put the data for lemons
X_data[200:,:] = dist_02

print(X_data)
print(labels)

# Knn algorithm we are done with the preparation of the dataset

# Distance of query point to all other points in the space
# O(N) time for every point + sorting time 
# Complexity for n points is o(Q.N) 

# We need a function to compute distance between 2 points in space
# These points could be one-d or 2-d or n-d representing a vector

# This gives the euclidean distance between the two points passed in the form of vectors
def distance(x1,x2):
    return np.sqrt(((x1-x2)**2).sum())

print(distance(np.array([0.0,0.0]),np.array([1.0,1.0])))


# Let us define the KNN 
# X_train is our training data(dataset)
# query_point is the one whose membership we are trying to determine
# y_train is the labels
# k is 5 which is the nearest neighbours which is generally odd
# k should not be too high or too low

def knn(X_train,query_point,y_train,k=5):
    vals=[]
    # Iterate over each of the points in the dataset
    for ix in range(X_train.shape[0]):
        # Compute the distance of the query point from all other points in the dataset
        v = [distance(query_point,X_train[ix,:]), y_train[ix]]
        #  Append these points to the vals list
        vals.append(v)
        # Now sort the list of all the vals
        updated_vals = sorted(vals)
        # Pick up the top k values(talking about the top k points)
        pred_arr = np.asarray(updated_vals[:k])
        
        # Find unique labels
        pred_arr = np.unique(pred_arr[:,1],return_counts= True)
        # Largest occurence
        index = pred_arr[1].argmax()   #Index of largest frequency
        return pred_arr[0][index]
       
q = np.array([1.0,2.0])
predicted_label = knn(X_data,q,labels)
print(predicted_label)


# Run a loop over the testing data(split it one into the training data and one for the testing data)
#  