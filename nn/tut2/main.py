import numpy as np
import pandas as pd

#q2
f = lambda x: x**2*np.sin(2*np.pi*x) + 0.7

def generate_random_point_and_class():
    x1 = np.random.rand()
    x2 = np.random.rand()
    _class = 0 if f(x1) > x2 else 1 
    
    return np.array([x1, x2]), _class

# print(generate_random_point_and_class())

def generate_sample_set():
    points = []
    labels = []
    for i in range(100):
        point_class = generate_random_point_and_class()
        points.append(point_class[0])
        labels.append(point_class[1])
        
    return np.array(points), np.array(labels)
    
X, Y = generate_sample_set()
    
#init weights
weights1 = np.random.normal(size=(3, 2))
biases1 = np.ones((3, 1))
weights2 = np.random.normal(size=(1, 3))
biases2 = 1
W = [weights1, weights2]
B = [biases1, biases2]

def forward_propagation(X, W, B):
    a = None
    for i, x in enumerate(X):
        #first layer
        w = W[0].T
        b = B[0]
        print(x.shape, w.shape)
        l1 = np.matmul(x, w) + b[0]
        print(l1)
forward_propagation(X, W, B) 

