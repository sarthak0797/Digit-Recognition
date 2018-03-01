import numpy as np
import mnist_loader as ml
import network as net
import test as tt

np.random.seed(1)

weights = 2*np.random.random((784,30)) - 1
weights1 = 2*np.random.random((30,10)) - 1

tr_data, val_data, test_data = ml.load_data()

tr_inputs = [np.reshape(x, (784, 1)) for x in tr_data[0]]
tr_outputs = [ml.vectorized_result(x) for x in tr_data[1]]

for i in range(10000):

    weights , weights1 = net.train(tr_inputs[i],tr_outputs[i],weights,weights1)
    if(i % 500) == 0 :
        print (i/500, "%")

print ("Network Trained and ready to be operated")

te_inputs = [np.reshape(x, (784,1)) for x in test_data[0]]
te_outputs = test_data[1]

tt.check(te_inputs,te_outputs,weights,weights1)
