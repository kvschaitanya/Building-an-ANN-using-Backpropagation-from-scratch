import numpy

class NAGOptimizer:
	def __init__(self, learning_rate = 0.01, momentum = 0.8):
		self.learning_rate = learning_rate
		self.momentum = momentum
		self.velocities = {}

	def update(self, layer, dw, db):
		if layer not in self.velocities:
			self.velocities[layer] = {
				'vw' : numpy.zeros(layer.weights.shape),
				'vb' : numpy.zeros(layer.bias.shape)
			}
		vw_prev = self.velocities[layer]['vw'].copy()
		vb_prev = self.velocities[layer]['vb'].copy()

		self.velocities[layer]['vw'] = self.momentum * vw_prev - self.learning_rate * dw
		self.velocities[layer]['vb'] = self.momentum * vb_prev - self.learning_rate * db

		layer.weights += self.velocities[layer]['vw'] + self.momentum * (self.velocities[layer]['vw'] - vw_prev)
		layer.bias += self.velocities[layer]['vb'] + self.momentum * (self.velocities[layer]['vb'] - vb_prev)

class GradientDescent:
	def __init__(self, learning_rate):
		self.learning_rate = learning_rate
	def update(self, layer, dw, db):
		layer.weights -= self.learning_rate * dw
		layer.bias -= self.learning_rate * db