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

def ANC(image,L1,L2):#Artificial Noise Comparison
	return [0,rmse(image,cv2.GaussianBlur(image,(L1*2+1,L1*2+1),0))]

def SVMClassifier(L1,L2):

	#title="EME("+str(L1)+","+str(L1)+") Std("+str(L2)+","+str(L2)+")"
	title='Test16'

	if os.path.exists('dataset_cache\\'+title+'.txt'):
		with open('dataset_cache\\'+title+'.txt','rb') as f:
			dataset=pickle.load(f)
	else:
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

		print("Excelentes:")
		for path in list_excelente:
			img=cv2.imread(path)
			parameter=ANC(img,L1,L1)
			print(path,':',parameter)
			parameters_excelente.append(parameter)
		print("Bons:")
		for path in list_bom:
			img=cv2.imread(path)
			parameter=ANC(img,L1,L1)
			print(path,':',parameter)
			parameters_bom.append(parameter)
		print("Ruins:")
		for path in list_ruim:
			img=cv2.imread(path)
			parameter=ANC(img,L1,L1)
			print(path,':',parameter)
			parameters_ruim.append(parameter)
		print("Pessimos:")
		for path in list_pessimo:
			img=cv2.imread(path)
			parameter=ANC(img,L1,L1)
			print(path,':',parameter)
			parameters_pessimo.append(parameter)

		X_excelente = np.asarray(parameters_excelente)
		X_bom = np.asarray(parameters_bom)
		X_ruim = np.asarray(parameters_ruim)
		X_pessimo = np.asarray(parameters_pessimo)

		X = np.append(X_excelente,X_bom,axis=0)
		X = np.append(X,X_ruim,axis=0)
		X = np.append(X,X_pessimo,axis=0)
		Y = np.append(np.ones((1,len(parameters_excelente)))*3,np.ones((1,len(parameters_bom)))*2)
		Y = np.append(Y,np.ones((1,len(parameters_ruim)))*1)
		Y = np.append(Y,np.ones((1,len(parameters_pessimo)))*0)
		dataset=[X,Y]
		with open('dataset_cache\\'+title+'.txt', 'wb') as f:
		    pickle.dump(dataset, f)

	for i in range(0,len(dataset[1])):
		if dataset[1][i]==1:
			dataset[1][i]=0
		elif dataset[1][i]==2:
			dataset[1][i]=1
		elif dataset[1][i]==3:
			dataset[1][i]=2
	n_classes=3
	sum=0
	tabela=np.zeros((n_classes,n_classes))
	for i in range(0,10000):
		X_train, X_test, y_train, y_test = train_test_split(dataset[0], dataset[1])
		clf = SVC(kernel='linear',class_weight='balanced')
		clf.fit(X_train, y_train)
		predictions = clf.predict(X_test)
		sum=sum+accuracy_score(y_test, predictions)
		for i in range(0,len(predictions)):
			tabela[int(predictions[i])][int(y_test[i])]=tabela[int(predictions[i])][int(y_test[i])]+1

	tabela=tabela/10000
	#tabela=np.append([['Péssimo'],['Ruim'],['Bom'],['Excelente']],tabela,axis=-1)
	#tabela=np.append([['Predição\Realidade','Péssimo','Ruim','Bom','Excelente']],tabela,axis=0)
	tabela=np.append([['Péssimo ou Ruim'],['Bom'],['Excelente']],tabela,axis=-1)
	tabela=np.append([['Predição\Realidade','Péssimo ou Ruim','Bom','Excelente']],tabela,axis=0)
	avg_accuracy=sum/10000
	print("Average accurary=",avg_accuracy)
	X_train, X_test, y_train, y_test = train_test_split(dataset[0], dataset[1])

	clf = SVC(kernel='linear',class_weight='balanced')
	clf.fit(X_train, y_train)

	predictions = clf.predict(X_test)
	
	x_visual=[]
	y_visual=[]

	for i in range (0,n_classes-1):
		w=clf.coef_[i]
		b=clf.intercept_[i]
		if w[0]==0:
			x_visual.append(np.linspace(-1,1))	
		else:
			x_visual.append(np.linspace(dataset[0].max(0)[0],dataset[0].min(0)[0]))
		y_visual.append(-(w[0] / w[1]) * x_visual[0] - b / w[1])
	

	
	colormap=[]
	for i in dataset[1]:
		if i==0:
			colormap.append('red')
		elif i==1:
			colormap.append('orange')
		elif i==2:
			colormap.append('green')
		elif i==3:
			colormap.append('blue')

	#class_legend=[["Péssimo","red"],["Ruim","orange"],["Bom","green"],["Excelente","blue"]]
	class_legend=[["Péssimo ou Ruim","red"],["Bom","orange"],["Excelente","green"]]

	#ax=plt.axes(projection="3d")

	for item in class_legend:
		#ax.scatter3D([],[],[],c=item[1],label=item[0])
		plt.scatter([],[],c=item[1],label=item[0])
	plt.legend()


	plt.title(title)


	#ax.scatter3D(X_train[:,0],X_train[:,1],X_train[:,2],c=colormap)
	plt.scatter(dataset[0][:,0],dataset[0][:,1],c=colormap)

	for i in range (0,n_classes-1):
		plt.plot(x_visual[i], y_visual[i],c=class_legend[i][1],label=class_legend[i][0]+'/'+class_legend[i+1][0]+' Divisor')
	plt.legend()
	print(tabulate(tabela))
	#plt.savefig(title)
	plt.show()
	#np.savetxt('matriz_confusao\\Fase_2\\'+title+'.csv', tabela, delimiter =", ",fmt="%s")
	

SVMClassifier(15,15)

