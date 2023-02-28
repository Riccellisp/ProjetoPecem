import tensorflow as tf
import csv
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import cv2
import os
import numpy as np


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
		img_resize=cv2.resize(img,(224,224),interpolation = cv2.INTER_AREA)
		if line[17]=='Excelente':
			X.append(img_resize)
			Y.append(3)
		elif line[17]=='Bom':
			X.append(img_resize)
			Y.append(2)
		elif line[17]=='Ruim':
			X.append(img_resize)
			Y.append(1)
		elif line[17]=='Pessimo':
			X.append(img_resize)
			Y.append(0)
X=np.asarray(X)
Y=np.asarray(Y)

dataset=[X,Y]

Xtrain, Xtest, Ytrain, Ytest = train_test_split(dataset[0], dataset[1])

model = tf.keras.applications.MobileNet(
	include_top=True,
    input_shape=(224,224,3),
    weights="imagenet",
    classes=1000
)

model = tf.keras.models.Model(inputs=model.input, outputs=model.get_layer(index=-2).output)

prediction = np.array(model.predict(Xtrain))
print(prediction.shape)
Xtrain = np.reshape(prediction, (prediction.shape[0],prediction.shape[1]))

prediction = np.array(model.predict(Xtest))
Xtest = np.reshape(prediction, (prediction.shape[0],prediction.shape[1]))

svm = SVC(kernel='linear')
svm.fit(Xtrain, np.ravel(Ytrain,order='C'))
result = svm.predict(Xtest)

acc = accuracy_score(result, np.ravel(Ytest, order='C'))

print("Accuracy = %0.4f" % acc)