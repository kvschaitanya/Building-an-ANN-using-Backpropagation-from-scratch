﻿# Building-an-ANN-using-Backpropagation-from-scratch

This project implements a basic artificial neural network from scratch in Python. It includes modularized components for different activation functions, layers, error functions, and optimizers, allowing flexibility to experiment with various configurations.

## Contents

- `Activation_functions.py`: Contains common activation functions (ReLU, Sigmoid, Tanh) and their derivatives.
- `Error_functions.py`: Implements Mean Squared Error (MSE) loss and its derivative.
- `Layer.py`: Contains classes for Dense layers with L2 regularization and Activation layers.
- `ANN.py`: The main neural network class that combines layers, error functions, and optimizers to train models.
- `Optimizer.py`: Includes implementations of Nesterov Accelerated Gradient (NAG) and basic Gradient Descent optimizers.

## Setup 💻

```bash
git clone <url>
pip install -r requirements.txt
```

### Usage
```python
import numpy as np
import matplotlib.pyplot as plt
from ANN import Network
from Layer import DenseLayer, ActivationLayer
from Activation_functions import tanh, d_tanh
from Error_functions import MSE, d_MSE
from Optimizer import NAGOptimizer

X = np.linspace(-2 * np.pi, 2 * np.pi, 1000).reshape((-1, 1))
y = np.sin(X)

nn = Network()
nn.add(DenseLayer(1, 20))
nn.add(ActivationLayer(tanh, d_tanh))
nn.add(DenseLayer(20, 20))
nn.add(ActivationLayer(tanh, d_tanh))
nn.add(DenseLayer(20, 1))
nn.add(ActivationLayer(tanh, d_tanh))

nn.compile(MSE, d_MSE, NAGOptimizer(0.0001))

history = nn.fit(X, y, 5000)
plt.plot(history)
plt.show()
plt.plot(X[:, 0], y[:, 0])
plt.plot(X[:, 0], nn.predict(X)[:, 0])
plt.show()
```
#### Output:
<pre>
  epoch 500/5000		error = 0.020351774143418974
  epoch 1000/5000		error = 0.013095934537433997
  epoch 1500/5000		error = 0.006614144833602568
  epoch 2000/5000		error = 0.0035421493601542338
  epoch 2500/5000		error = 0.0018434027661203495
  epoch 3000/5000		error = 0.0010365681940266854
  epoch 3500/5000		error = 0.0006635661351849564
  epoch 4000/5000		error = 0.0005190628301680765
  epoch 4500/5000		error = 0.00044838439658660055
  epoch 5000/5000		error = 0.0004026510215804483
</pre>

<p align="center">
  <img src="https://github.com/user-attachments/assets/146d24a6-46f1-41e7-a5e9-c1e4941bff7e" alt="Error plot"><br>
  Error
</p>

<p align="center">
  <img src = "https://github.com/user-attachments/assets/d0dc7f81-2ee2-4724-ba7a-e77f67bd6b04" alt = "Predicted vs Actual Sine Wave"><br>
  Predicted vs Actual Sine Wave
</p>
