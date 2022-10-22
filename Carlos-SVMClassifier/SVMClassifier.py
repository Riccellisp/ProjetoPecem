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



'''
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
	parameter=[eme(img,5,5),emee(img,5,5)]
	parameters_excelente.append(parameter)

with open('excelenteParameters.txt', 'wb') as f:
    pickle.dump(parameters_excelente, f)

for path in list_bom:
	img=cv2.imread(path)
	parameter=[eme(img,5,5),emee(img,5,5)]
	parameters_bom.append(parameter)

with open('bomParameters.txt', 'wb') as f:
    pickle.dump(parameters_bom, f)

for path in list_ruim:
	img=cv2.imread(path)
	parameter=[eme(img,5,5),emee(img,5,5)]
	parameters_ruim.append(parameter)

with open('ruimParameters.txt', 'wb') as f:
    pickle.dump(parameters_ruim, f)

for path in list_pessimo:
	img=cv2.imread(path)
	parameter=[eme(img,5,5),emee(img,5,5)]
	parameters_pessimo.append(parameter)

with open('pessimoParameters.txt', 'wb') as f:
    pickle.dump(parameters_pessimo, f)
'''


with open('excelenteParameters.txt','rb') as f:
	parameters_excelente=pickle.load(f)

with open('bomParameters.txt','rb') as f:
	parameters_bom=pickle.load(f)

with open('ruimParameters.txt','rb') as f:
	parameters_ruim=pickle.load(f)

with open('pessimoParameters.txt','rb') as f:
	parameters_pessimo=pickle.load(f)

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

sum=0
for i in range(0,50):
	X_train, X_test, y_train, y_test = train_test_split(X, Y)
	clf = SVC(kernel='linear')
	clf.fit(X_train, y_train)
	predictions = clf.predict(X_test)
	sum=sum+accuracy_score(y_test, predictions)

avg_accuracy=sum/50
print('Average accuracy = ',avg_accuracy)


X_train, X_test, y_train, y_test = train_test_split(X, Y)

clf = SVC(kernel='poly')
clf.fit(X_train, y_train)

predictions = clf.predict(X_test)
print('Accuracy = ', accuracy_score(y_test, predictions))
'''
n_classes=4

x_visual=[]
y_visual=[]

for i in range (0,n_classes-1):
	w=clf.coef_[i]
	b=clf.intercept_[i]
	x_visual.append(np.linspace(0,20))
	y_visual.append(-(w[0] / w[1]) * x_visual[i] - b / w[1])
'''
colormap=[]
for i in y_train:
	if i==0:
		colormap.append('red')
	elif i==1:
		colormap.append('orange')
	elif i==2:
		colormap.append('green')
	elif i==3:
		colormap.append('blue')
	
	

plt.scatter(X_train[:,0],X_train[:,1],c=colormap)
'''
for i in range (0,n_classes-1):
	plt.plot(x_visual[i], y_visual[i])
'''

plt.show()

