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
from standardDesv import standardDesv
from tabulate import tabulate
from gridsearch_knn import gridsearchKNN
from sklearn.neighbors import KNeighborsClassifier

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
Y = np.append(np.ones((1,len(parameters_excelente)))*2,np.ones((1,len(parameters_bom)))*1)
Y = np.append(Y,np.ones((1,len(parameters_ruim)))*0)
Y = np.append(Y,np.ones((1,len(parameters_pessimo)))*0)
n_classes=3
sum=0
tabela=np.zeros((3,3))

for i in range(0,50):
    X_train, X_test, y_train, y_test = train_test_split(X, Y)
    X_train_1,X_val,y_train_1,y_val = train_test_split(X_train, y_train)
    parametros = {'k':[3, 5, 7],'distance':['euclidean','cityblock']}
    resultGrid = gridsearchKNN(parametros, X_train_1,y_train_1,X_val, y_val) 
    
    print(resultGrid)
    clf = KNeighborsClassifier(n_neighbors=resultGrid['k'], metric=resultGrid['distance'])

    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)
    sum=sum+accuracy_score(y_test, predictions)
    for i in range(0,len(predictions)):
        tabela[int(predictions[i])][int(y_test[i])]=tabela[int(predictions[i])][int(y_test[i])]+1

for i in range(0,n_classes):
	tabela[i]=tabela[i]/np.sum(tabela[i])
avg_accuracy=sum/50
print('Average accuracy = ',avg_accuracy)
header=["Predição\Realidade",'0','1','2']
print(tabulate(tabela,headers=header,tablefmt="fancy_grid", showindex="always"))

X_train, X_test, y_train, y_test = train_test_split(X, Y)

clf = SVC(kernel='linear')
clf.fit(X_train, y_train)

predictions = clf.predict(X_test)
print('Accuracy = ', accuracy_score(y_test, predictions))

x_visual=[]
y_visual=[]

for i in range (0,n_classes-1):
	w=clf.coef_[i]
	b=clf.intercept_[i]
	x_visual.append(np.linspace(10,20))
	y_visual.append(-(w[0] / w[1]) * x_visual[i] - b / w[1])

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

for i in range (0,n_classes-1):
	plt.plot(x_visual[i], y_visual[i])


plt.show()

