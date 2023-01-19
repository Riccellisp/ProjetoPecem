import cv2
import numpy as np
import random
import os
import math
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

def augmentation(image):
	image=changeBrightness(image,0.8,1.2)
	image=horizontalFlip(image)
	image=verticalFlip(image)
	image=rotation(image,30)
	return image

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

def modifiedVariance_of_laplacian(image,L1,L2):
	resized = cv2.resize(image,(1680,860),interpolation = cv2.INTER_AREA)
	grayImg = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
	blurImg = cv2.GaussianBlur(grayImg,(L1-L1%2+1,L2-L2%2+1),0)
	rowSize, columnSize = grayImg.shape
	nRows = int(rowSize/L1)
	nColumns = int(columnSize/L2)
	responseVector=[]

	for i in range(0,nRows):
		for j in range(0,nColumns):
			responseVector.append(cv2.Laplacian(grayImg[i*L1:(i+1)*L1,j*L2:(j+1)*L1], cv2.CV_64F).var())
	return responseVector

def modifiedStd(image,L1,L2):
	resized = cv2.resize(image,(1680,860),interpolation = cv2.INTER_AREA)
	grayImg = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
	blurImg = cv2.GaussianBlur(grayImg,(L1-L1%2+1,L2-L2%2+1),0)
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
			responseVector.append(math.sqrt(meanSqrtError))
	return responseVector

def metric1(image):
	#return variance_of_laplacian(image)
	#return eme(image,20,20)
	#return emee(image,50,50)
	#return standardDesv(image,10,10)
	#return ANC(image,20,20)
	return modifiedANC(image,20,20)
	#return modifiedVariance_of_laplacian(image,20,20)
	#return modifiedStd(image,20,20)
def metric2(image):
	#return variance_of_laplacian(image)
	#return eme(image,20,20)
	#return emee(image,50,50)
	#return standardDesv(image,10,10)
	return ANC(image,20,20)


def SVMClassifier():
	title='Novo dataset'
	n_classes=2
	
	pathDatabase=os.getcwd()+'\\dataset_pecem'

	pathExcelente=pathDatabase+'\\Excelente'
	pathBom=pathDatabase+'\\Bom'
	pathRuim=pathDatabase+'\\Ruim'
	pathPessimo=pathDatabase+'\\Pessimo'

	x_path=[]
	y=[]

	for root, dirs, files in os.walk(pathExcelente):
		for file in files:
			x_path.append(os.path.join(root,file))
			y.append(1)
	
	for root, dirs, files in os.walk(pathBom):
		for file in files:
			x_path.append(os.path.join(root,file))
			y.append(1)

	for root, dirs, files in os.walk(pathRuim):
		for file in files:
			x_path.append(os.path.join(root,file))
			y.append(0)

	for root, dirs, files in os.walk(pathPessimo):
		for file in files:
			x_path.append(os.path.join(root,file))
			y.append(0)

	x_train_path, x_test_path, y_train_preaug, y_test = train_test_split(x_path, y,test_size=0.25)

	X_train=[]
	y_train=[]
	X_test=[]

	for i in range(0,len(x_train_path)):
		image=cv2.imread(x_train_path[i])
		if image is not None:
			for i in range(0,20):
				img_aug=augmentation(image)
				parameter=metric1(img_aug)
				X_train.append(parameter)
				y_train.append(y_train_preaug[i])
		else:
			print("Path error!")

	for path in x_test_path:
		img=cv2.imread(path)
		if img is not None:
			parameter=metric1(img)
			X_test.append(parameter)
		else:
			print("Path error!")
	
	X_train=np.asarray(X_train)
	X_test=np.asarray(X_test)
	y_train=np.asarray(y_train)
	y_test=np.asarray(y_test)
	#print("X_train: ",X_train)
	#print("X_test: ",X_test)
	#print("y_train: ",y_train)
	#print("y_test: ",y_test)
	sum=0
	tabela=np.zeros((n_classes,n_classes))

	
	#X_train, X_test, y_train, y_test = train_test_split(dataset[0], dataset[1])
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

SVMClassifier()

