import numpy as np


def normalize(X):
    mu = np.mean(X, 0)
    sigma = np.std(X, 0)
    Xnorm = (X - mu) / sigma
    return (Xnorm, mu, sigma)
