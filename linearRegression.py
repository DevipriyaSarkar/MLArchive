import numpy as np

data=np.genfromtxt("data1.txt",delimiter=',')
# second arg to hsplit is number of pieces or array of indices
X,Y=np.hsplit(data,[-1])
X=np.hstack((np.ones((len(X),1)),X))

def cost(X,Y,theta,rlambda):
	m = len(Y)
	grad = np.zeros(theta.shape)
	err = np.dot(X,theta) - Y
	J = ((1 / (2 * m)) * np.power(err,2).sum()
		# regulirazation term
		+ (rlambda / (2 * m)) * np.power(theta[1:],2).sum())
	grad = ((1 / m) * (np.dot(X.T,err))
		# regulirazation term
		+ (rlambda / m) * (np.vstack((0,theta[1:]))))

	return (J,grad)

theta = np.zeros((2,1))
rlambda = 0;
print(cost(X,Y,theta,rlambda))

