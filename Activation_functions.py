import numpy
import matplotlib.pyplot as plt

def tanh(x):
    return numpy.tanh(x)

def d_tanh(x):
    return 1 - numpy.tanh(x) ** 2

def sigmoid(x):
	return 1 / (1 + numpy.exp(-x))

def d_sigmoid(x):
	return sigmoid(x) * (1 - sigmoid(x))

def relu(x):
	return numpy.maximum(0, x)

def d_relu(x):
	return numpy.where(x > 0, 1, 0)

if __name__ == "__main__":
	plt.figure(figsize = (12, 9))

	plt.subplot(3, 2, 1)
	plt.plot(numpy.linspace(-5, 5, 100), tanh(numpy.linspace(-5, 5, 100)))
	plt.title("tanh function")
	plt.ylabel("tanh(x)")
	plt.xlabel("x")

	plt.subplot(3, 2, 2)
	plt.plot(numpy.linspace(-5, 5, 100), d_tanh(numpy.linspace(-5, 5, 100)))
	plt.title("derivative of tanh")
	plt.ylabel("tanh'(x)")
	plt.xlabel("x")

	plt.subplot(3, 2, 3)
	plt.plot(numpy.linspace(-10, 10, 100), sigmoid(numpy.linspace(-10, 10, 100)))
	plt.title("sigmoid function")
	plt.ylabel("sigmoid(x)")
	plt.xlabel("x")

	plt.subplot(3, 2, 4)
	plt.plot(numpy.linspace(-10, 10, 100), d_sigmoid(numpy.linspace(-10, 10, 100)))
	plt.title("derivative of sigmoid")
	plt.ylabel("sigmoid'(x)")
	plt.xlabel("x")

	plt.subplot(3, 2, 5)
	plt.plot(numpy.linspace(-50, 50, 100), relu(numpy.linspace(-50, 50, 100)))
	plt.title("relu function")
	plt.ylabel("relu(x)")
	plt.xlabel("x")

	plt.subplot(3, 2, 6)
	plt.plot(numpy.linspace(-50, 50, 100), d_relu(numpy.linspace(-50, 50, 100)))
	plt.title("derivative of relu")
	plt.ylabel("relu'(x)")
	plt.xlabel("x")

	plt.show()