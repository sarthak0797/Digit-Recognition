import Network as net
import numpy as np


def feedforward(x,weights,weights1):

    l = x.T
    l1 = net.sigmoid(np.dot(l,weights))
    l2 = net.sigmoid(np.dot(l1,weights1))
    return l2;

def check(te_inputs,te_outputs,weights,weights1):

    correct = 0
    
    for i in range(len(te_inputs)):
        
        out = feedforward(te_inputs[i],weights,weights1)
        f_out = np.argmax(out)
        if(f_out == te_outputs[i]):
            correct += 1

    print ("Accuracy Of the Network is " , ((correct/10000)*100))
