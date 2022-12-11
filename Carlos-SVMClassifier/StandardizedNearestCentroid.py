import numpy as np

class StandardizedNearestCentroid:
	def __init__(self):
		self.classes=[]
	def fit(self,x_train,y_train):
		label_classes=list(set(y_train))
		for label in label_classes:
			aux=[]
			for j in range(0,len(y_train)):
				if y_train[j]==label:
					#print(x_train[j])
					aux.append(x_train[j])
			aux=np.asarray(aux)
			self.classes.append([label,np.mean(aux,axis=0),np.sqrt(np.sum(np.square(aux-np.mean(aux,axis=0))))])
	def predict(self,x_test):
		prediction=[]
		for point in x_test:
			smallest_distance=None
			best_guess=None
			for guess in self.classes:
				distance=np.sqrt(np.sum(np.square(point-guess[1])))/guess[2]
				if smallest_distance==None or distance<smallest_distance:
					smallest_distance=distance
					best_guess=guess
			prediction.append(best_guess[0])
		return prediction
	def test_predict(self,x_test,y_test):
		prediction=[]
		right=0
		wrong=0
		for point in x_test:
			smallest_distance=None
			best_guess=None
			for guess in self.classes:
				distance=np.sqrt(np.sum(np.square(point-guess[1])))/guess[2]
				if smallest_distance==None or distance<smallest_distance:
					smallest_distance=distance
					best_guess=guess[0]
			prediction.append(best_guess)
		for i in range(0,len(prediction)):
			if(prediction[i]==y_test[i]):
				right=right+1
			else:
				wrong=wrong+1
		return right/(right+wrong) 


