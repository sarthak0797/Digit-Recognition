# Digit-Recognition
  This projects trains an ANN with one hidden layer of 30 neurons and a CNN using backpropogation algorithm to recognise handwritten digits.

# Data-Set-Used
  I have used the MNIST data set to train the network for 10,000 images.
  You can learn more about the data set here http://yann.lecun.com/exdb/mnist/.
  Basically the MNIST data set contains a set of 60,000 training images and it also contains 10,000 test images of 28 X 28 pixels which we     have converted into a matrix of 784 X 1
  The data set can be cloned or downloaded from the DATA SET directory or you can download it from this link           https://drive.google.com/drive/folders/1RkvokHfxD4WkRHwPJ0H9dcjRyc8r6X7z
  
# Backpropagation
  Backpropagation is a method used in artificial neural networks to calculate a gradient that is needed in the calculation of the weights to be used in the network. 
  It is commonly used to train deep neural networks , a term used to explain neural networks with more than one hidden layer.

  Backpropagation is a special case of an older and more general technique called automatic differentiation. In the context of learning, backpropagation is commonly used by the gradient descent optimization algorithm to adjust the weight of neurons by calculating the gradient of the loss function. This technique is also sometimes called backward propagation of errors, because the error is calculated at the output and distributed back through the network layers.
  The backpropagation algorithm was originally introduced in the 1970s, but its importance wasn't fully appreciated until a famous 1986 paper by David Rumelhart, Geoffrey Hinton, and Ronald Williams. That paper describes several neural networks where backpropagation works far faster than earlier approaches to learning, making it possible to use neural nets to solve problems which had previously been insoluble. Today, the backpropagation algorithm is the workhorse of learning in neural networks.
  
  
# Analysis
  The ANN achieved an accuracy of 92.5%.
  
  The CNN achieved an accuracy of 98.7%
  
# How to run
  Run Digit_Recognition_ANN.py for Artifical Neural Network.
  
  Run Digit_Recognition_CNN.py for Convountional Neural Network
  
  
 
