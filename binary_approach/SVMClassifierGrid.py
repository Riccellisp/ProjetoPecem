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
from gridsearch_svm import gridsearchSVM

pathDatabase=os.getcwd()+'\\dataset_pecem_binary'
pathBom=pathDatabase+'\\Bom'
pathRuim=pathDatabase+'\\Ruim'

list_bom=[]
list_ruim=[]

for root, dirs, files in os.walk(pathBom):
	for file in files:
		list_bom.append(os.path.join(root,file))

for root, dirs, files in os.walk(pathRuim):
	for file in files:
		list_ruim.append(os.path.join(root,file))


parameters_bom=[]
parameters_ruim=[]



for path in list_bom:
	img=cv2.imread(path)
	parameter=[eme(img,20,20),standardDesv(img,20,20)]
	parameters_bom.append(parameter)

with open('bomParameters.txt', 'wb') as f:
    pickle.dump(parameters_bom, f)

for path in list_ruim:
	img=cv2.imread(path)
	parameter=[eme(img,20,20),standardDesv(img,20,20)]
	parameters_ruim.append(parameter)

with open('ruimParameters.txt', 'wb') as f:
    pickle.dump(parameters_ruim, f)




with open('bomParameters.txt','rb') as f:
	parameters_bom=pickle.load(f)

with open('ruimParameters.txt','rb') as f:
	parameters_ruim=pickle.load(f)

X_bom = np.asarray(parameters_bom)
X_ruim = np.asarray(parameters_ruim)

Y = []

X = np.append(X_bom,X_ruim,axis=0)
Y = np.append(Y,np.ones((1,len(parameters_bom))))
Y = np.append(Y,np.ones((1,len(parameters_ruim)))*0)

n_classes=2
sum=0
tabela=np.zeros((2,2))

for i in range(0,50):
    X_train, X_test, y_train, y_test = train_test_split(X, Y)
    X_train_1,X_val,y_train_1,y_val = train_test_split(X_train, y_train)
    parametros = {'C':[1, 10, 20],'kernel':['rbf', 'poly','sigmoid']}
    resultGrid = gridsearchSVM(parametros, X_train_1,y_train_1,X_val, y_val) 
    print(resultGrid)
    clf = SVC(kernel=resultGrid['kernel'], C=resultGrid['C'])

    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)
    sum=sum+accuracy_score(y_test, predictions)
    for i in range(0,len(predictions)):
        tabela[int(predictions[i])][int(y_test[i])]=tabela[int(predictions[i])][int(y_test[i])]+1

for i in range(0,n_classes):
	tabela[i]=tabela[i]/np.sum(tabela[i])
avg_accuracy=sum/50
print('Average accuracy = ',avg_accuracy)
header=["Predição\Realidade",'0','1']
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

