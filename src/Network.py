import numpy as np

"""Return the sigmoid value or activation value of our neuron i:e it maps any value to a value between 0 to 1"""
def sigmoid(x):
    return 1/(1+np.exp(-x))

"""Return the derivative of the signoid function"""
def deriv_sigmoid(x):
    return x*(1-x)

def train(inputs,output,weights,weights1):
    
    x = inputs.T
    y = output.T
    """Giving the input to our network and calculating the outptut and
        storing it in l2"""
    
    l1 = sigmoid(np.dot(x,weights))
    l2 = sigmoid(np.dot(l1,weights1))

    """Calculating the error by subtracting our Networks output from expected output"""
    error = y - l2 

    """This gives how much did our output layer contributed in our missed output"""
    l2_del = error * deriv_sigmoid(l2)

    
    """Calculating the error of out hidden layer"""
    error0 = l2_del.dot(weights1.T)

    """This gives how much did our hidden layer contributed in our missed output"""
    l1_del = error0 * deriv_sigmoid(l1)

    """updating the values of our weights by how much we missed"""
    weights1 += np.dot(l1.T,l2_del)
    weights += np.dot(x.T,l1_del)

    return weights,weights1


