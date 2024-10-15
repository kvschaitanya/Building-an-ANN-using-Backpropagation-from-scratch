# Building-an-ANN-using-Backpropagation-from-scratch

This project implements a basic artificial neural network from scratch in Python. It includes modularized components for different activation functions, layers, error functions, and optimizers, allowing flexibility to experiment with various configurations.

## Project Structure

- `Activation_functions.py`: Contains common activation functions (ReLU, Sigmoid, Tanh) and their derivatives.
- `Error_functions.py`: Implements Mean Squared Error (MSE) loss and its derivative.
- `Layer.py`: Contains classes for Dense layers with L2 regularization and Activation layers.
- `ANN.py`: The main neural network class that combines layers, error functions, and optimizers to train models.
- `Optimizer.py`: Includes implementations of Nesterov Accelerated Gradient (NAG) and basic Gradient Descent optimizers.