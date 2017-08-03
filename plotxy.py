import  matplotlib.pyplot as plt
import numpy as np

if __name__=="__main__":
	x=np.arange(1,10)
	y=2*x+1
	fig=plt.figure()
	ax=fig.add_subplot(1,1,1)
	ax.scatter(x,y,color='red',marker='o')
	ax.plot(x,y,color='lightblue',linewidth=1)
	plt.show()