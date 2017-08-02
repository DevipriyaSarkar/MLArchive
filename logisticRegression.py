import numpy as np


def sigmoid(z):
	""" Computes the sigmoid of z """
	return 1/(1+np.exp(-z))


if __name__ == "__main__":
	print(sigmoid(np.array([(-100),(0),(100)])))