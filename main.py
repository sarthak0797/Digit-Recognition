import numpy as np
import mnist_loader as ml

class network:

    def __init__(self):

        np.random.seed(1)

        self.sy = 2*np.random.random((784,10)) - 1
        self.sy1 = 2*np.random.random((10,10)) - 1


    def vectorized_result(self,j):
        
        e = np.zeros((10, 1))
        e[j] = 1.0
        return e

    def nonlin(self,x,deriv = False):
        if(deriv == True):
            return x*(1-x)
        return 1/(1+np.exp(-x))

    def train(self,inputs,output):
        
        x = inputs.T
        y = output.T

        for i in range(1):

            l = x
            l1 = self.nonlin(np.dot(l,self.sy))
            l2 = self.nonlin(np.dot(l1,self.sy1))
            
            error = y - l2

            l2_del = error * self.nonlin(l2,True)

            error0 = l2_del.dot(self.sy1.T)

            l1_del = error0 * self.nonlin(l1,True)

            self.sy1 += np.dot(l1.T,l2_del)
            self.sy += np.dot(l.T,l1_del)

        return self.sy,self.sy1

    def output(self,l2):

        print ("Result After Training is ")
        print (l2)

    def checker(self,x):

        l = x.T
        l1 = self.nonlin(np.dot(l,self.sy))
        l2 = self.nonlin(np.dot(l1,self.sy1))
        return l2;


net = network()

tr_data, val_data, test_data = ml.load_data()

tr_inputs = [np.reshape(x, (784, 1)) for x in tr_data[0]]
tr_outputs = [net.vectorized_result(x) for x in tr_data[1]]

for i in range(10000):

    weight1 , weight2 = net.train(tr_inputs[i],tr_outputs[i])
    if(i % 100) == 0 :
        print (i/100, "%")

print ("Network Trained and ready to be operated")

te_inputs = [np.reshape(x, (784,1)) for x in test_data[0]]
te_outputs = test_data[1]

correct = 0

for i in range(len(te_inputs)):
    
    out = net.checker(te_inputs[i])
    f_out = np.argmax(out)
    if(f_out == te_outputs[i]):
        correct += 1

print ("Accuracy Of the Network is " , ((correct/10000)*100))

#net.output(l2)



