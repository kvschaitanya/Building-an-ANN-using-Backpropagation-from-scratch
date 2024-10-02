class Network:
	def __init__(self):
		self.layers = []
		self.error = None
		self.d_error = None
	def add(self, layer):
		self.layers.append(layer)
	def compile(self, error, d_error):
		self.error = error
		self.d_error = d_error
	def fit(self, X, y, epochs, learning_rate = 0.01):
		loss_history = []
		for _ in range(epochs):
			output = X.copy()
			for i in range(len(self.layers)):
				output = self.layers[i].forward_propagation(output)

			output_error = self.d_error(output, y)
				  
			for i in range(len(self.layers) - 1, -1, -1):
				output_error = self.layers[i].backward_propagation(output_error, learning_rate)
				
			loss_history.append(self.error(output, y))
		return loss_history
				
	def predict(self, x):
		output = x.copy()
		for layer in self.layers:
			output = layer.forward_propagation(output)
		return output