import tensorflow as tf
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import cv2
import os
import numpy as np

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
		parameter=cv2.resize(img,(224,224))
		parameters_excelente.append(parameter)
	else:
		print(path)

for path in list_bom:
	img=cv2.imread(path)
	if img is not None:
		parameter=cv2.resize(img,(224,224))
		parameters_bom.append(parameter)
	else:
		print(path)

for path in list_ruim:
	img=cv2.imread(path)
	if img is not None:
		parameter=cv2.resize(img,(224,224))
		parameters_ruim.append(parameter)
	else:
		print(path)

for path in list_pessimo:
	img=cv2.imread(path)
	if img is not None:
		parameter=cv2.resize(img,(224,224))
		parameters_pessimo.append(parameter)
	else:
		print(path)

X = np.array(parameters_excelente+parameters_bom+parameters_ruim+parameters_pessimo,dtype=float)/255
Y=[]
for i in parameters_excelente:
	Y.append(3)
for i in parameters_bom:
	Y.append(2)
for i in parameters_ruim:
	Y.append(1)
for i in parameters_pessimo:
	Y.append(0)
Y=np.array(Y,dtype=float)
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