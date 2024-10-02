import numpy

class DenseLayer:
	def __init__(self, input_size, output_size):
		self.weights = numpy.random.random((input_size, output_size)) - 0.5
		self.bias = numpy.random.random((1, output_size)) - 0.5

	def forward_propagation(self, input):
		self.input = input
		self.output = input @ self.weights + self.bias
		return self.output

	def backward_propagation(self, output_error, learning_rate):
		dy_dw = self.input.T @ output_error
		dy_dx = output_error @ self.weights.T
		self.weights -= learning_rate * dy_dw
		self.bias -= learning_rate * output_error.sum(axis = 0)
		return dy_dx
	
class ActivationLayer:
	def __init__(self, activation, activation_d):
		self.activation = activation
		self.activation_d = activation_d
	def forward_propagation(self, input):
		self.input = input
		return self.activation(input)
	def backward_propagation(self, output_error, learning_rate):
		return output_error * self.activation_d(self.input)
