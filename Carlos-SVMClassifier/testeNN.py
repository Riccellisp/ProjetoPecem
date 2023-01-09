import numpy as np
import os
import cv2
import math
import tensorflow as tf
import torch
import random
from sklearn.model_selection import train_test_split

def standardDesv(img,rowSample,columnSample):
	
	resized = cv2.resize(img,(1680,860),interpolation = cv2.INTER_AREA)
	grayImg = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
	rowSize, columnSize = grayImg.shape
	nRows = int(rowSize/rowSample)
	nColumns = int(columnSize/columnSample)
	responseMatrix=np.zeros((nRows,nColumns))

	for i in range(0,nRows):
		for j in range(0,nColumns):
			responseMatrix[i,j]=np.std(grayImg[i*rowSample:(i+1)*rowSample,j*columnSample:(j+1)*rowSample])
	return np.reshape(responseMatrix,(1,nRows,nColumns,1))

def ANC(img,rowSample,columnSample):
	resized = cv2.resize(img,(1680,860),interpolation = cv2.INTER_AREA)
	grayImg = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
	blurImg = cv2.GaussianBlur(grayImg,(rowSample-rowSample%2+1,columnSample-columnSample%2+1),0)
	rowSize, columnSize = grayImg.shape
	nRows = int(rowSize/rowSample)
	nColumns = int(columnSize/columnSample)
	responseMatrix=np.zeros((nRows,nColumns))

	for i in range(0,nRows):
		for j in range(0,nColumns):
			error = np.subtract(grayImg[i*rowSample:(i+1)*rowSample,j*columnSample:(j+1)*rowSample],
				blurImg[i*rowSample:(i+1)*rowSample,j*columnSample:(j+1)*rowSample])
			sqrtError = np.square(error)
			meanSqrtError = np.mean(sqrtError)
			responseMatrix[i,j] = math.sqrt(meanSqrtError)
	#return np.reshape(responseMatrix,(1,nRows,nColumns,1))
	return np.expand_dims(responseMatrix,axis=(0,3))

def generate_dataset(rowSample,columnSample):

	pathDatabase=os.getcwd()+'\\dataset_pecem_augmented'

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
			parameter=ANC(img,rowSample,columnSample)
			parameters_excelente.append(parameter)
		else:
			print(path)

	for path in list_bom:
		img=cv2.imread(path)
		if img is not None:
			parameter=ANC(img,rowSample,columnSample)
			parameters_bom.append(parameter)
		else:
			print(path)

	for path in list_ruim:
		img=cv2.imread(path)
		if img is not None:
			parameter=ANC(img,rowSample,columnSample)
			parameters_ruim.append(parameter)
		else:
			print(path)

	for path in list_pessimo:
		img=cv2.imread(path)
		if img is not None:
			parameter=ANC(img,rowSample,columnSample)
			parameters_pessimo.append(parameter)
		else:
			print(path)

	X = np.array(parameters_excelente+parameters_bom+parameters_ruim+parameters_pessimo,dtype=float)/255
	print("X Shape = ",X.shape)
	Y=[]
	for i in parameters_excelente:
		Y.append([0,0,0,1])
	for i in parameters_bom:
		Y.append([0,0,1,0])
	for i in parameters_ruim:
		Y.append([0,1,0,0])
	for i in parameters_pessimo:
		Y.append([1,0,0,0])
	Y=np.array(Y,dtype=float)
	print("Y Shape = ",Y.shape)
	dataset=[X,Y]

	return dataset

dataset = generate_dataset(40,40)

n_x, batch, input_size1, input_size2, channel = dataset[0].shape
n_y, output_size = dataset[1].shape
X_train, X_test, y_train, y_test = train_test_split(dataset[0], dataset[1])

model = tf.keras.Sequential()
model.add(tf.keras.Input(shape=(batch, input_size1, input_size2, channel)))
model.add(tf.keras.layers.Conv2D(filters=1,kernel_size=3,input_shape=(batch, input_size1, input_size2, channel),activation='relu'))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(4,activation='softmax'))
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=[tf.keras.metrics.CategoricalAccuracy()])
es_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=3)
losses = model.fit(X_train,y_train,
		validation_data=(X_test,y_test),
		callbacks=[es_callback],
		epochs=200
		)

print(losses)