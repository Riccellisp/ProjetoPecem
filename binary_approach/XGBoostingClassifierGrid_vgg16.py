# -*- coding: utf-8 -*-
"""
Created on Thu Dec  1 15:19:00 2022

@author: lueli
"""

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
import xgboost as xgb



from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
import random
from tabulate import tabulate
from gridsearch_gb import gridsearchXGB
from vgg16_feature_extraction import vgg16_feature_extraction

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
    parametros = {'gamma': [0,0.1,0.2,0.4,0.8,1.6,3.2,6.4,12.8,25.6,51.2,102.4, 200],
                  'learning_rate': [0.01, 0.03, 0.06, 0.1, 0.15, 0.2, 0.25, 0.300000012, 0.4, 0.5, 0.6, 0.7],
                  'max_depth': [5,6,7,8,9,10,11,12,13,14],
                  'n_estimators': [50,65,80,100,115,130,150],
                  'reg_alpha': [0,0.1,0.2,0.4,0.8,1.6,3.2,6.4,12.8,25.6,51.2,102.4,200],
                  'reg_lambda': [0,0.1,0.2,0.4,0.8,1.6,3.2,6.4,12.8,25.6,51.2,102.4,200]}
    resultGrid = gridsearchXGB(parametros, X_train_1,y_train_1,X_val, y_val) 
    print(resultGrid)
    clf = xgb.XGBClassifier(gamma=resultGrid['gamma'],learning_rate=resultGrid['learning_rate'],max_depth=resultGrid['max_depth'],n_estimators=resultGrid['n_estimators'],reg_alpha=resultGrid['reg_alpha'],reg_lambda=resultGrid['reg_lambda'])

    
    
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
header=["Predi????o\Realidade",'0','1']
print(tabulate(tabela,headers=header,tablefmt="fancy_grid", showindex="always"))
