import cv2
import numpy as np
import random
import os
from eme import eme
from emee import emee
import matplotlib.pyplot as plt
import pickle
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import random
from standardDesv import standardDesv,stdSpecial,emeSpecial,rmse
from tabulate import tabulate
from sklearn.neighbors import NearestCentroid
from sklearn.inspection import DecisionBoundaryDisplay
from matplotlib.colors import ListedColormap

def IEM_filter(image):
	image = image.astype(np.float32)

	(iH, iW) = image.shape[:2]
	output = np.zeros((iH-2, iW-2), dtype="float32")

	for y in np.arange(0, iH-2):
		for x in np.arange(0, iW-2):

			roi = image[y:y + 3, x:x + 3]

			k = roi
			k = (np.abs(k[1][1] - k[0][0]) + np.abs(k[1][1] - k[0][1]) + np.abs(k[1][1] - k[0][2]) +
				np.abs(k[1][1] - k[1][0]) + np.abs(k[1][1] - k[1][2]) +
				np.abs(k[1][1] - k[2][0]) + np.abs(k[1][1] - k[2][1]) + np.abs(k[1][1] - k[2][2]))  

			output[y, x] = k
	return np.sum(output)

def IEM(imageA, imageB):
	'''
	Image Enhancement Metric(IEM) approximates the contrast and
	sharpness of an image by dividing an image into non-overlapping
	blocks. 

	imageA is the raw image (before histogram equalization per example), and imageB 
	is the image after preprocessing
	'''
	valA = IEM_filter(imageA)
	valB = IEM_filter(imageB)

	return valB/valA

def variance_of_laplacian(image):
	'''
	compute the Laplacian of the image and then return the focus
	measure, which is simply the variance of the Laplacian
	''' 
	return cv2.Laplacian(image, cv2.CV_64F).var()

def ANC(image,L1,L2):#Artificial Noise Comparison
	return rmse(image,cv2.GaussianBlur(image,(L1*2+1,L1*2+1),0))

def metric1(image):
	#return variance_of_laplacian(image)
	#return eme(image,20,20)
	#return emee(image,50,50)
	#return standardDesv(image,10,10)
	return ANC(image,20,20)
def metric2(image):
	#return variance_of_laplacian(image)
	#return eme(image,20,20)
	#return emee(image,50,50)
	#return standardDesv(image,10,10)
	return ANC(image,20,20)

def NCCClassifier():
	title='Teste'
	n_classes=3
	
	pathDatabase=os.getcwd()+'\\dataset_pecem'

	pathExcelente=pathDatabase+'\\Excelente'
	pathBom=pathDatabase+'\\Bom'
	pathRuim=pathDatabase+'\\Ruim'
	pathPessimo=pathDatabase+'\\Pessimo'

	list_excelente=[]
	list_bom=[]
	list_ruim=[]
	list_pessimo=[]


	for root, dirs, files in os.walk(pathExcelente):
		for file in files:
			list_excelente.append(os.path.join(root,file))
	
	for root, dirs, files in os.walk(pathBom):
		for file in files:
			list_bom.append(os.path.join(root,file))

	for root, dirs, files in os.walk(pathRuim):
		for file in files:
			list_ruim.append(os.path.join(root,file))

	for root, dirs, files in os.walk(pathPessimo):
			for file in files:
				list_pessimo.append(os.path.join(root,file))

	parameters_excelente=[]
	parameters_bom=[]
	parameters_ruim=[]
	parameters_pessimo=[]
	
	for path in list_excelente:
		img=cv2.imread(path)
		if img is not None:
			parameter=[0,metric2(img)]
			#parameter=metric1(img)
			parameters_excelente.append(parameter)
		else:
			print(path)

	for path in list_bom:
		img=cv2.imread(path)
		if img is not None:
			parameter=[0,metric2(img)]
			#parameter=metric1(img)
			parameters_bom.append(parameter)
		else:
			print(path)

	for path in list_ruim:
		img=cv2.imread(path)
		if img is not None:
			parameter=[0,metric2(img)]
			#parameter=metric1(img)
			parameters_ruim.append(parameter)
		else:
			print(path)

	for path in list_pessimo:
		img=cv2.imread(path)
		if img is not None:
			parameter=[0,metric2(img)]
			#parameter=metric1(img)
			parameters_pessimo.append(parameter)
		else:
			print(path)

	X_excelente = np.asarray(parameters_excelente)
	X_bom = np.asarray(parameters_bom)
	X_ruim = np.asarray(parameters_ruim)
	X_pessimo = np.asarray(parameters_pessimo)
	X = X_excelente
	X = np.append(X,X_bom,axis=0)
	X = np.append(X,X_ruim,axis=0)
	X = np.append(X,X_pessimo,axis=0)
	Y = np.ones((1,len(parameters_excelente)))*3
	Y = np.append(Y,np.ones((1,len(parameters_bom)))*2)
	Y = np.append(Y,np.ones((1,len(parameters_ruim)))*1)
	Y = np.append(Y,np.ones((1,len(parameters_pessimo)))*0)

	dataset=[X,Y]

	if n_classes==3:
		for i in range(0,len(dataset[1])):
			if dataset[1][i]==1:
				dataset[1][i]=0
			elif dataset[1][i]==2:
				dataset[1][i]=1
			elif dataset[1][i]==3:
				dataset[1][i]=2
	if n_classes==2:
		for i in range(0,len(dataset[1])):
			if dataset[1][i]==1:
				dataset[1][i]=0
			elif dataset[1][i]==3:
				dataset[1][i]=2
	sum=0
	tabela=np.zeros((n_classes,n_classes))

	for i in range(0,1000):
		X_train, X_test, y_train, y_test = train_test_split(dataset[0], dataset[1],train_size=0.8)
		clf = NearestCentroid()
		clf.fit(X_train, y_train)
		predictions = clf.predict(X_test)
		sum=sum+accuracy_score(y_test, predictions)
		for i in range(0,len(predictions)):
			tabela[int(predictions[i])][int(y_test[i])]=tabela[int(predictions[i])][int(y_test[i])]+1

	tabela=tabela/1000
	if n_classes==4:
		tabela=np.append([['Péssimo'],['Ruim'],['Bom'],['Excelente']],tabela,axis=-1)
		tabela=np.append([['Predição\Realidade','Péssimo','Ruim','Bom','Excelente']],tabela,axis=0)
		class_legend=[["Péssimo","red"],["Ruim","orange"],["Bom","green"],["Excelente","blue"]]
		np.savetxt('matriz_confusao\\4 Classes\\'+title+'.csv', tabela, delimiter =", ",fmt="%s")
	elif n_classes==3:
		tabela=np.append([['Ruim'],['Bom'],['Excelente']],tabela,axis=-1)
		tabela=np.append([['Predição\Realidade','Ruim','Bom','Excelente']],tabela,axis=0)
		class_legend=[["Ruim","red"],["Bom","orange"],["Excelente","green"]]
		np.savetxt('matriz_confusao\\3 Classes\\'+title+'.csv', tabela, delimiter =", ",fmt="%s")
	elif n_classes==2:
		tabela=np.append([['Ruim'],['Excelente']],tabela,axis=-1)
		tabela=np.append([['Predição\Realidade','Ruim','Excelente']],tabela,axis=0)
		class_legend=[["Ruim","red"],["Excelente","green"]]
		np.savetxt('matriz_confusao\\2 Classes\\'+title+'.csv', tabela, delimiter =", ",fmt="%s")
	avg_accuracy=sum/1000
	print("Average accuracy=",avg_accuracy)
	print(tabulate(tabela))
	X_train, X_test, y_train, y_test = train_test_split(dataset[0], dataset[1],train_size=0.8)

	clf = NearestCentroid()
	clf.fit(X_train, y_train)

	predictions = clf.predict(X_test)

	if n_classes==4:
		cmap=ListedColormap(["red","orange","green","blue"])

		DecisionBoundaryDisplay.from_estimator(clf, X_train, cmap=cmap, response_method="predict")

		plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cmap, edgecolor="k", s=20)
		plt.title("Nearest Centroid 4-Class classification")
	if n_classes==3:
		cmap=ListedColormap(["red","orange","green"])

		DecisionBoundaryDisplay.from_estimator(clf, X_train, cmap=cmap, response_method="predict")

		plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cmap, edgecolor="k", s=20)
		plt.title("Nearest Centroid 3-Class classification")

    #plt.axis("tight")

	if(n_classes==4):
		plt.savefig('Gráfico dispersão\\4 Classes\\'+title+'.png')
	elif(n_classes==3):
		plt.savefig('Gráfico dispersão\\3 Classes\\'+title+'.png')
	elif(n_classes==2):
		plt.savefig('Gráfico dispersão\\3 Classes\\'+title+'.png')
	plt.show()
	

NCCClassifier()

