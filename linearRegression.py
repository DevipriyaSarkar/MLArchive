import numpy as np


def cost(X, Y, theta, rlambda):
    """ Computes linear regression cost and gradient of using theta 
    to fit data X and Y with regularization parameter rlambda """
    m = len(Y)
    grad = np.zeros(theta.shape)
    err = np.dot(X, theta) - Y
    J = ((1 / (2 * m)) * np.dot(err.T, err)
         # regulirazation term
         + (rlambda / (2 * m)) * np.dot(theta[1:].T, theta[1:]))
    grad = ((1 / m) * (np.dot(X.T, err))
            # regulirazation term
            + (rlambda / m) * (np.vstack((0, theta[1:]))))

    return (J, grad)


def gradientDescent(X, Y, rlambda, alpha, num_iter):
    """ Performs gradient descent with num_iter iterations, 
    learning rate alpha and regularization term rlambda 
    returns the learned theta """
    theta = np.zeros((X.shape[1], 1))
    for i in range(num_iter):
        (J, grad) = cost(X, Y, theta, rlambda)
        # here we can use J to plot wrt i to check convergence and select alpha
        theta = theta - alpha * grad
    return theta


if __name__ == "__main__":
    data = np.genfromtxt("data1.txt", delimiter=',')
    # second arg to hsplit is number of pieces or array of indices
    # split X and Y from data
    X, Y = np.hsplit(data, [-1])
    # add X0 = 1
    X = np.hstack((np.ones((len(X), 1)), X))
    theta = np.zeros((2, 1))
    rlambda = 0
    print(cost(X,Y,theta,rlambda))
    print(gradientDescent(X, Y, rlambda, 0.01, 1500))
