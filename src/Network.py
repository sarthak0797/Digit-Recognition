import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))

def deriv_sigmoid(x):
    return x*(1-x)

def train(inputs,output,weights,weights1):
    
    x = inputs.T
    y = output.T

    for i in range(1):

        l = x
        l1 = sigmoid(np.dot(l,weights))
        l2 = sigmoid(np.dot(l1,weights1))
        
        error = y - l2

        l2_del = error * deriv_sigmoid(l2)

        error0 = l2_del.dot(weights1.T)

        l1_del = error0 * deriv_sigmoid(l1)

        weights1 += np.dot(l1.T,l2_del)
        weights += np.dot(l.T,l1_del)

    return weights,weights1

def checker(x,weights,weights1):

    l = x.T
    l1 = sigmoid(np.dot(l,weights))
    l2 = sigmoid(np.dot(l1,weights1))
    return l2;


