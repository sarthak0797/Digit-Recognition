import tensorflow as tf
import numpy as np
import mnist_loader as ml
import bar as br
import random
import gzip
import pickle

f = gzip.open('mnist.pkl.gz', 'rb')

training_data, validation_data, test_data = pickle.load(f, encoding = 'latin1')

f.close()

tr_inputs = [np.reshape(x, (1,784)) for x in training_data[0]]
tr_outputs = [ml.vectorized_result(x) for x in training_data[1]]


te_inputs = [np.reshape(x, (1,784)) for x in test_data[0]]
te_outputs = [ml.vectorized_result(x) for x in test_data[1]]

n_classes = 10
batch_size = 256

x = tf.placeholder('float', [None, 784])
y = tf.placeholder('float')

def conv2d(x ,W):
	return tf.nn.conv2d(x, W, strides = [1,1,1,1], padding = 'SAME')

def maxpool2d(x):
	return tf.nn.max_pool(x, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME')

def convulation_net(x):
	weights = {'W_conv1' : tf.Variable(tf.random_normal([5,5,1,32])),
			   'W_conv2' : tf.Variable(tf.random_normal([5,5,32,64])),
			   'W_fc' : tf.Variable(tf.random_normal([7*7*64,1024])),
			   'out' : tf.Variable(tf.random_normal([1024,n_classes])),
			   }

	biases = {'b_conv1' : tf.Variable(tf.random_normal([32])),
			  'b_conv2' : tf.Variable(tf.random_normal([64])),
			  'b_fc' : tf.Variable(tf.random_normal([1024])),
			  'out' : tf.Variable(tf.random_normal([n_classes])),
			  }		  
	
	x = tf.reshape(x, shape = [-1, 28, 28, 1])

	conv1 = tf.nn.relu(conv2d(x, weights['W_conv1']) + biases['b_conv1'])
	conv1 = maxpool2d(conv1)

	conv2 = tf.nn.relu(conv2d(conv1, weights['W_conv2']) + biases['b_conv2'])
	conv2 = maxpool2d(conv2)


	fc = tf.reshape(conv2, [-1,7*7*64])
	fc = tf.nn.relu(tf.matmul(fc,weights['W_fc']) + biases['b_fc'])

	ouptut = tf.matmul(fc, weights['out']) + biases['out']

	return ouptut

def train_net(x):

	#sess = tf.InteractiveSession
	prediction = convulation_net(x)
	print(prediction)
	cost = tf.nn.softmax_cross_entropy_with_logits(logits = prediction, labels=y)
	optimizer = tf.train.AdamOptimizer(1e-4).minimize(cost)
	correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

	with tf.Session() as sess :
		sess.run(tf.global_variables_initializer())

		for i in range(20000):

			if(i % 200) == 0 :
				br.progress(i, 20000)

			for j in range(10):
				k = random.randint(0,20000)
				optimizer.run(feed_dict = {x: tr_inputs[k], y: tr_outputs[k]})

		br.progress(20000, 20000, cond = True)

		print("\n")

		acc = 0
		for i in range(10000):
			acc += accuracy.eval(feed_dict={
				x: te_inputs[i], y: te_outputs[i]})

		print('test accuracy %g' % acc)




train_net(x)