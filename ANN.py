class Network:
	def __init__(self):
		self.layers = []
		self.error = None
		self.d_error = None
	def add(self, layer):
		self.layers.append(layer)
	def compile(self, error, d_error, optimizer):
		self.error = error
		self.d_error = d_error
		self.optimizer = optimizer
	def fit(self, X, y, epochs):
		loss_history = []
		interval = epochs / 10
		for epoch in range(epochs):
			output = X.copy()
			for i in range(len(self.layers)):
				output = self.layers[i].forward_propagation(output)

			output_error = self.d_error(output, y)
				  
			for i in range(len(self.layers) - 1, -1, -1):
				output_error = self.layers[i].backward_propagation(output_error, self.optimizer)
				
			loss_history.append(self.error(output, y))
			if (epoch + 1) % interval == 0:
				print(f"epoch {epoch + 1}/{epochs}\t\terror = {self.error(output, y)}")

		return loss_history
				
	def predict(self, x):
		output = x.copy()
		for layer in self.layers:
			output = layer.forward_propagation(output)
		return output