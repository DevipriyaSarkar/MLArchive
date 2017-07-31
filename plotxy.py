import  matplotlib.pyplot as plt

def plotxy(X,Y):
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.scatter(X,Y,color='red',marker='x')
	plt.show()