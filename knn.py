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

