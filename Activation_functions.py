import numpy

def tanh(x):
    return (numpy.exp(x) - numpy.exp(-x)) / (numpy.exp(x) + numpy.exp(-x))

def d_tanh(x):
    return 1 - tanh(x) ** 2

def sigmoid(x):
	return 1 / (1 + numpy.exp(-x))

def d_sigmoid(x):
	return sigmoid(x) * (1 - sigmoid(x))

def relu(x):
	return numpy.maximum(0, x)

def d_relu(x):
	return numpy.where(x > 0, 1, 0)