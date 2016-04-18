import numpy as np

"""sigmoid function"""
def sigmoid(value):
	if value < -500:
		value = -500
	if value > 500:
		value = 500
	return 1.0 / (1.0 + np.exp(-value))

"""hyperpol function"""
def hyperpol(value):
	return (1.0 - np.exp(-value)) / (1.0 + np.exp(-value))

"""tang-hyperpol function"""
def tanh(value):
	return 2.0 / (1 - np.exp(-2.0 * value)) - 1.0

def derivativeSigmoid(value):
	return sigmoid(value) * (1.0 - sigmoid(value))

def derivativeTanh(value):
	return 1.0 - tanh(value) * tanh(value)