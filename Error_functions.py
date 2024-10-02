import numpy

def MSE(y_pred, y_actual):
	return numpy.mean(((y_pred - y_actual) ** 2) * 0.5)

def d_MSE(y_pred, y_actual) -> numpy.ndarray:
	return (y_pred - y_actual)