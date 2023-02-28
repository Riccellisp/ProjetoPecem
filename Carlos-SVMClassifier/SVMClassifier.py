import cv2
import numpy as np
import random
import os
import math
import csv
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

def changeBrightness(img,low,high):
	    value = random.uniform(low,high)
	    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
	    hsv = np.array(hsv, dtype = np.float64)
	    hsv[:,:,1] = hsv[:,:,1]*value
	    hsv[:,:,1][hsv[:,:,1]>255]  = 255
	    hsv[:,:,2] = hsv[:,:,2]*value 
	    hsv[:,:,2][hsv[:,:,2]>255]  = 255
	    hsv = np.array(hsv, dtype = np.uint8)
	    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
	    return img

def horizontalFlip(img):
	return cv2.flip(img,int(random.getrandbits(1)))

def verticalFlip(img):
	return cv2.flip(img,int(random.getrandbits(1)))

def rotation(img,max_angle):
    angle = int(random.uniform(-max_angle, max_angle))
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((int(w/2), int(h/2)), angle, 1)
    img = cv2.warpAffine(img, M, (w, h))
    return img

def augmentation(img):
	img=changeBrightness(img,0.95,1.05)
	img=horizontalFlip(img)
	img=verticalFlip(img)
	img=rotation(img,10)
	return img

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
	return [0,rmse(image,cv2.GaussianBlur(image,(L1*2+1,L1*2+1),0))]

def modifiedVariance_of_laplacian(image,L1,L2):
	resized = cv2.resize(image,(1680,860),interpolation = cv2.INTER_AREA)
	grayImg = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
	rowSize, columnSize = grayImg.shape
	nRows = int(rowSize/L1)
	nColumns = int(columnSize/L2)
	responseVector=[]

	for i in range(0,nRows):
		for j in range(0,nColumns):
			responseVector.append(cv2.Laplacian(grayImg[i*L1:(i+1)*L1,j*L2:(j+1)*L1], cv2.CV_64F).var())
	return responseVector

def modifiedEme(image,L1,L2):
	resized = cv2.resize(image,(1680,860),interpolation = cv2.INTER_AREA)
	grayImg = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
	rowSize, columnSize = grayImg.shape
	nRows = int(rowSize/L1)
	nColumns = int(columnSize/L2)
	responseVector=[]

	for i in range(0,nRows):
		for j in range(0,nColumns):
			maximo=np.amax(grayImg[i*L1:(i+1)*L1,j*L2:(j+1)*L1])
			minimo=np.amin(grayImg[i*L1:(i+1)*L1,j*L2:(j+1)*L1])
			if maximo==0:
				responseVector.append(0)
			elif minimo==0:
				responseVector.append(20*math.log(maximo/0.5))
			else:
				responseVector.append(20*math.log(maximo/minimo))
	return responseVector

def modifiedEmee(image,L1,L2):
	resized = cv2.resize(image,(1680,860),interpolation = cv2.INTER_AREA)
	grayImg = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
	rowSize, columnSize = grayImg.shape
	nRows = int(rowSize/L1)
	nColumns = int(columnSize/L2)
	responseVector=[]

	for i in range(0,nRows):
		for j in range(0,nColumns):
			maximo=np.amax(grayImg[i*L1:(i+1)*L1,j*L2:(j+1)*L1])
			minimo=np.amin(grayImg[i*L1:(i+1)*L1,j*L2:(j+1)*L1])
			if maximo==0:
				responseVector.append(0)
			elif minimo==0:
				responseVector.append(maximo/0.5*math.log(maximo/0.5))
			else:
				responseVector.append(maximo/minimo*math.log(maximo/minimo))
	return responseVector

def modifiedStd(image,L1,L2):
	resized = cv2.resize(image,(1680,860),interpolation = cv2.INTER_AREA)
	grayImg = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
	rowSize, columnSize = grayImg.shape
	nRows = int(rowSize/L1)
	nColumns = int(columnSize/L2)
	responseVector=[]

	for i in range(0,nRows):
		for j in range(0,nColumns):
			responseVector.append(np.std(grayImg[i*L1:(i+1)*L1,j*L2:(j+1)*L1]))
	return responseVector

def modifiedANC(image,L1,L2):
	resized = cv2.resize(image,(1680,860),interpolation = cv2.INTER_AREA)
	grayImg = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
	blurImg = cv2.GaussianBlur(grayImg,(L1-L1%2+1,L2-L2%2+1),0)
	rowSize, columnSize = grayImg.shape
	nRows = int(rowSize/L1)
	nColumns = int(columnSize/L2)
	responseVector=[]

	for i in range(0,nRows):
		for j in range(0,nColumns):
			error = np.subtract(grayImg[i*L1:(i+1)*L1,j*L2:(j+1)*L1],
				blurImg[i*L1:(i+1)*L1,j*L2:(j+1)*L1])
			sqrtError = np.square(error)
			meanSqrtError = np.mean(sqrtError)
			responseVector.append(math.sqrt(meanSqrtError)/255)
	return responseVector

def PSNR(image,L1,L2):
	resized = cv2.resize(image,(1680,860),interpolation = cv2.INTER_AREA)
	grayImg = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
	blurImg = cv2.GaussianBlur(grayImg,(L1-L1%2+1,L2-L2%2+1),0)
	rowSize, columnSize = grayImg.shape
	nRows = int(rowSize/L1)
	nColumns = int(columnSize/L2)
	responseVector=[]

	for i in range(0,nRows):
		for j in range(0,nColumns):
			error = np.subtract(grayImg[i*L1:(i+1)*L1,j*L2:(j+1)*L1],
				blurImg[i*L1:(i+1)*L1,j*L2:(j+1)*L1])
			sqrtError = np.square(error)
			meanSqrtError = np.mean(sqrtError)
			if meanSqrtError>0:
				responseVector.append(10*math.log(255*255/meanSqrtError)/math.log(10))
			else:
				responseVector.append(99)
	return responseVector

def metric1(image):
	#return modifiedEme(image,20,20)
	#return modifiedEmee(image,20,20)
	#return modifiedANC(image,40,40)
	#return modifiedVariance_of_laplacian(image,40,40)
	#return modifiedStd(image,20,20)
	return PSNR(image,20,20)
def metric2(image):
	#return variance_of_laplacian(image)
	#return eme(image,20,20)
	#return emee(image,50,50)
	#return standardDesv(image,10,10)
	return ANC(image,20,20)


def SVMClassifier():
	title='Novo dataset'
	n_classes=4

	tabela=[]
	with open('teste_pecem.csv','r') as csvfile:
	    spamreader = csv.reader(csvfile)
	    for line in spamreader:
	   		tabela.append(line)

	X=[]
	Y=[]

	for line in tabela:
		img=cv2.imread(line[1])
		if img is not None:
			#parameter=metric1(img)
			if line[20]=='Excelente':
				X.append(line[1])
				Y.append(3)
			elif line[20]=='Bom':
				X.append(line[1])
				Y.append(2)
			elif line[20]=='Ruim':
				X.append(line[1])
				Y.append(1)
			elif line[20]=='Pessimo':
				X.append(line[1])
				Y.append(0)
	X=np.asarray(X)
	Y=np.asarray(Y)

	dataset=[X,Y]

	sum=0
	tabela=np.zeros((n_classes,n_classes))

	#################

	X_train, X_test, y_train, y_test = train_test_split(dataset[0], dataset[1])
	new_X_train=[]
	new_y_train=[]
	new_X_test=[]
	new_y_test=[]
	training_number=[0,0,0,0]
	for i in range(0,len(y_train)):
		img=cv2.imread(X_train[i])
		if y_train[i]==0:
			for k in range(0,16):
				img_aug=augmentation(img)
				new_X_train.append(metric1(img_aug))
				new_y_train.append(0)
				training_number[0]+=1
		elif y_train[i]==1:
			for k in range(0,6):
				img_aug=augmentation(img)
				new_X_train.append(metric1(img_aug))
				new_y_train.append(1)
				training_number[1]+=1
		elif y_train[i]==2:
			for k in range(0,2):
				img_aug=augmentation(img)
				new_X_train.append(metric1(img_aug))
				new_y_train.append(2)
				training_number[2]+=1
		elif y_train[i]==3:
			for k in range(0,2):
				img_aug=augmentation(img)
				new_X_train.append(metric1(img_aug))
				new_y_train.append(3)
				training_number[3]+=1
	for i in range(0,len(y_test)):
		img=cv2.imread(X_test[i])
		if y_test[i]==0:
			new_X_test.append(metric1(img))
			new_y_test.append(0)
		elif y_test[i]==1:
			new_X_test.append(metric1(img))
			new_y_test.append(1)
		elif y_test[i]==2:
			new_X_test.append(metric1(img))
			new_y_test.append(2)
		elif y_test[i]==3:
			new_X_test.append(metric1(img))
			new_y_test.append(3)
	print("Number of training data: ",training_number)
	X_train=np.asarray(new_X_train,dtype=float)
	y_train=np.asarray(new_y_train,dtype=float)
	X_test=np.asarray(new_X_test,dtype=float)
	y_test=np.asarray(new_y_test,dtype=float)

	###############
	


	clf = SVC(kernel='linear',class_weight='balanced')
	clf.fit(X_train, y_train)
	predictions = clf.predict(X_test)
	sum=sum+accuracy_score(y_test, predictions)
	for i in range(0,len(predictions)):
		tabela[int(predictions[i])][int(y_test[i])]=tabela[int(predictions[i])][int(y_test[i])]+1

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
		tabela=np.append([['Limpa'],['Nao Limpa']],tabela,axis=-1)
		tabela=np.append([['Predição\Realidade','Limpa','Nao Limpa']],tabela,axis=0)
		class_legend=[["Limpa","red"],["Nao Limpa","green"]]
		np.savetxt('matriz_confusao\\2 Classes\\'+title+'.csv', tabela, delimiter =", ",fmt="%s")
	avg_accuracy=sum
	print("Average accuracy=",avg_accuracy)
	print(tabulate(tabela))

def NCCClassifier():
	title='Novo dataset'
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

	X_excelente = np.asarray(parameters_excelente)
	X_bom = np.asarray(parameters_bom)
	X_ruim = np.asarray(parameters_ruim)
	X = X_excelente
	X = np.append(X,X_bom,axis=0)
	X = np.append(X,X_ruim,axis=0)
	Y = np.ones((1,len(parameters_excelente)))*3
	Y = np.append(Y,np.ones((1,len(parameters_bom)))*2)
	Y = np.append(Y,np.ones((1,len(parameters_ruim)))*1)

	if n_classes==4:
		for root, dirs, files in os.walk(pathPessimo):
			for file in files:
				list_pessimo.append(os.path.join(root,file))
		for path in list_pessimo:
			img=cv2.imread(path)
			if img is not None:
				parameter=[0,metric2(img)]
				#parameter=metric1(img)
				parameters_pessimo.append(parameter)
			else:
				print(path)
		X_pessimo = np.asarray(parameters_pessimo)
		X = np.append(X,X_pessimo,axis=0)
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
		X_train, X_test, y_train, y_test = train_test_split(dataset[0], dataset[1])
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
	X_train, X_test, y_train, y_test = train_test_split(dataset[0], dataset[1])

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
	

SVMClassifier()

