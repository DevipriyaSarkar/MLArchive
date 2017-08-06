import numpy as np


def sigmoid(z):
    """ Computes the sigmoid of z """
    return 1 / (1 + np.exp(-z))


def cost(X, Y, theta, rlambda):
    """ Computes logistic regression cost and gradient of using theta
    to fit data X and Y with regularization parameter rlambda """
    m = len(Y)
    Hx = sigmoid(np.dot(X, theta))
    grad = np.zeros(theta.shape)
    J = ((1 / m) * (-np.dot(Y.T, np.log(Hx)) - np.dot(1 - Y.T, np.log(Hx)))
         + (rlambda / (2 * m)) * np.dot(theta[1:].T, theta[1:]))
    grad = ((1 / m) * (np.dot(X.T, Hx - Y))
            + (rlambda / m) * (np.vstack((0, theta[1:]))))
    return (J, grad)


if __name__ == "__main__":
    print(sigmoid(np.array([(-100), (0), (100)])))
    data = np.genfromtxt("data3.txt", delimiter=',')
    # second arg to hsplit is number of pieces or array of indices
    # split X and Y from data
    X, Y = np.hsplit(data, [-1])
    # add X0 = 1
    X = np.hstack((np.ones((len(X), 1)), X))
    theta = np.zeros((3, 1))
    rlambda = 0
    print(cost(X, Y, theta, rlambda))
