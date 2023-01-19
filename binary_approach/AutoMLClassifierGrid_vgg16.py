# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 02:36:20 2022

@author: lueli
"""

import cv2
import numpy as np
import random
import os
import matplotlib.pyplot as plt
import pickle
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
import random
from tabulate import tabulate
from gridsearch_automl import gridsearchAutoML
from vgg16_feature_extraction import vgg16_feature_extraction

try:
    import autosklearn.classification
except ImportError:
    import pip
    pip.main(['install', '--user', 'auto-sklearn'])
    import autosklearn.classification

pathDatabase=os.getcwd()+'\\dataset_pecem_binary'
pathBom=pathDatabase+'\\Bom'
pathRuim=pathDatabase+'\\Ruim'
'''
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
	parameter=vgg16_feature_extraction(img)
	parameters_bom.append(parameter)

with open('bomParametersVGG16.txt', 'wb') as f:
    pickle.dump(parameters_bom, f)

for path in list_ruim:
	img=cv2.imread(path)
	parameter=vgg16_feature_extraction(img)
	parameters_ruim.append(parameter)

with open('ruimParametersVGG16.txt', 'wb') as f:
    pickle.dump(parameters_ruim, f)
    
'''    
    
    
    
with open('bomParametersVGG16.txt','rb') as f:
	parameters_bom=pickle.load(f)

with open('ruimParametersVGG16.txt','rb') as f:
	parameters_ruim=pickle.load(f)    
    
    
    
X_bom = np.asarray(parameters_bom)
X_ruim = np.asarray(parameters_ruim)
Y = []

X = np.append(X_bom,X_ruim,axis=0)
Y = np.append(Y,np.ones((1,len(parameters_bom))))
Y = np.append(Y,np.ones((1,len(parameters_ruim)))*0)    

del X_bom
del X_ruim
del parameters_bom
del parameters_ruim

X = np.reshape(X,(40,4096))

n_classes=2
sum=0
tabela=np.zeros((2,2))

for i in range(0,500):
    X_train, X_test, y_train, y_test = train_test_split(X, Y)
    X_train_1,X_val,y_train_1,y_val = train_test_split(X_train, y_train)
    parametros = {'n_estimators':[10, 50,100],'max_depth':[None, 5,10]}
    resultGrid = gridsearchAutoML(parametros, X_train_1,y_train_1,X_val, y_val) 
    print(resultGrid)
    clf = autosklearn.classification.AutoSklearnClassifier(n_estimators =resultGrid['n_estimators'], max_depth = resultGrid['max_depth'] )    
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    
    clf.fit(X_train, y_train)
    X_test = scaler.transform(X_test)
    predictions = clf.predict(X_test)
    sum=sum+accuracy_score(y_test, predictions)
    for i in range(0,len(predictions)):
        tabela[int(predictions[i])][int(y_test[i])]=tabela[int(predictions[i])][int(y_test[i])]+1

for i in range(0,n_classes):
	tabela[i]=tabela[i]/np.sum(tabela[i])
avg_accuracy=sum/500
print('Average accuracy = ',avg_accuracy)
header=["Predição\Realidade",'0','1']
print(tabulate(tabela,headers=header,tablefmt="fancy_grid", showindex="always"))
