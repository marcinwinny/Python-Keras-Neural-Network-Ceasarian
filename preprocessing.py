import numpy
from numpy import loadtxt

# load the dataset
dataset = loadtxt('caesarian_dataset.arff', delimiter=',')

# split into input (X) and output (y) variables
X = numpy.array(dataset[:,0:5])
X[:,0] = X[:,0]/40
X[:,1] = X[:,1]/4
X[:,2] = X[:,2]/2
X[:,3] = X[:,3]/2
print(X)
y = dataset[:,5]