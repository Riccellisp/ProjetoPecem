import os
import tensorflow as tf
import cv2
import csv
import numpy as np
import math
import random
from tabulate import tabulate
from sklearn.model_selection import train_test_split

class ancCNN:
	
	def __init__(self):
		self.model=None
		self.rowSample=40
		self.columnSample=40
		self.y_train=[]
		self.y_test=[]

	def ANC(self,img):
		resized = cv2.resize(img,(1680,860),interpolation = cv2.INTER_AREA)
		grayImg = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
		blurImg = cv2.GaussianBlur(grayImg,(self.rowSample-self.rowSample%2+1,self.columnSample-self.columnSample%2+1),0)
		rowSize, columnSize = grayImg.shape
		nRows = int(rowSize/self.rowSample)
		nColumns = int(columnSize/self.columnSample)
		responseMatrix=np.zeros((nRows,nColumns))

		for i in range(0,nRows):
			for j in range(0,nColumns):
				error = np.subtract(grayImg[i*self.rowSample:(i+1)*self.rowSample,j*self.columnSample:(j+1)*self.rowSample],
					blurImg[i*self.rowSample:(i+1)*self.rowSample,j*self.columnSample:(j+1)*self.rowSample])
				sqrtError = np.square(error)
				meanSqrtError = np.mean(sqrtError)
				responseMatrix[i,j] = math.sqrt(meanSqrtError)
		return responseMatrix/255

	def STD(self,img):
		resized = cv2.resize(img,(1680,860),interpolation = cv2.INTER_AREA)
		grayImg = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
		blurImg = cv2.GaussianBlur(grayImg,(self.rowSample-self.rowSample%2+1,self.columnSample-self.columnSample%2+1),0)
		rowSize, columnSize = grayImg.shape
		nRows = int(rowSize/self.rowSample)
		nColumns = int(columnSize/self.columnSample)
		responseMatrix=np.zeros((nRows,nColumns))

		for i in range(0,nRows):
			for j in range(0,nColumns):
				responseMatrix[i,j]=np.std(grayImg[i*self.rowSample:(i+1)*self.rowSample,j*self.columnSample:(j+1)*self.rowSample])
		return responseMatrix

	def changeBrightness(self,img,low,high):
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

	def horizontalFlip(self,img):
		return cv2.flip(img,int(random.getrandbits(1)))

	def verticalFlip(self,img):
		return cv2.flip(img,int(random.getrandbits(1)))

	def rotation(self,img,max_angle):
	    angle = int(random.uniform(-max_angle, max_angle))
	    h, w = img.shape[:2]
	    M = cv2.getRotationMatrix2D((int(w/2), int(h/2)), angle, 1)
	    img = cv2.warpAffine(img, M, (w, h))
	    return img

	def augmentation(self,img):
		img=self.changeBrightness(img,0.95,1.05)
		img=self.horizontalFlip(img)
		img=self.verticalFlip(img)
		img=self.rotation(img,10)
		return img

	def generate_dataset(self,pathDatabase):

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
				#parameter=np.expand_dims(self.ANC(img),axis=(0,3))
				if line[20]=='Excelente':
					X.append(line[1])
					Y.append(3)
				elif line[20]=='Bom':
					X.append(line[1])
					Y.append(2)
				elif line[20]=='Ruim':
					X.append(line[1])
					Y.append(1)
				elif line[20]=='Pessimo':
					X.append(line[1])
					Y.append(0)
		X=np.asarray(X)
		Y=np.asarray(Y)

		dataset=[X,Y]

		X_train, X_test, y_train, y_test = train_test_split(dataset[0], dataset[1])
		new_X_train=[]
		new_y_train=[]
		new_X_test=[]
		new_y_test=[]
		training_number=[0,0,0,0]
		for i in range(0,len(y_train)):
			img=cv2.imread(X_train[i])
			if y_train[i]==0:
				for k in range(0,18):
					img_aug=self.augmentation(img)
					new_X_train.append(np.expand_dims(self.ANC(img_aug),axis=(0,3)))
					new_y_train.append([1,0,0,0])
					training_number[0]+=1
			elif y_train[i]==1:
				for k in range(0,6):
					img_aug=self.augmentation(img)
					new_X_train.append(np.expand_dims(self.ANC(img_aug),axis=(0,3)))
					new_y_train.append([0,1,0,0])
					training_number[1]+=1
			elif y_train[i]==2:
				for k in range(0,2):
					img_aug=self.augmentation(img)
					new_X_train.append(np.expand_dims(self.ANC(img_aug),axis=(0,3)))
					new_y_train.append([0,0,1,0])
					training_number[2]+=1
			elif y_train[i]==3:
				for k in range(0,2):
					img_aug=self.augmentation(img)
					new_X_train.append(np.expand_dims(self.ANC(img_aug),axis=(0,3)))
					new_y_train.append([0,0,0,1])
					training_number[3]+=1
		for i in range(0,len(y_test)):
			img=cv2.imread(X_test[i])
			if y_test[i]==0:
				new_X_test.append(np.expand_dims(self.ANC(img),axis=(0,3)))
				new_y_test.append([1,0,0,0])
			elif y_test[i]==1:
				new_X_test.append(np.expand_dims(self.ANC(img),axis=(0,3)))
				new_y_test.append([0,1,0,0])
			elif y_test[i]==2:
				new_X_test.append(np.expand_dims(self.ANC(img),axis=(0,3)))
				new_y_test.append([0,0,1,0])
			elif y_test[i]==3:
				new_X_test.append(np.expand_dims(self.ANC(img),axis=(0,3)))
				new_y_test.append([0,0,0,1])
		print("Number of training data: ",training_number)
		X_train=np.asarray(new_X_train,dtype=float)
		y_train=np.asarray(new_y_train,dtype=float)
		X_test=np.asarray(new_X_test,dtype=float)
		y_test=np.asarray(new_y_test,dtype=float)

		return X_train,X_test,y_train,y_test

	def train_model(self,pathDatabase):
		X_train,X_test,y_train,y_test=self.generate_dataset(pathDatabase)
		print(X_train.shape)
		print(y_train.shape)
		print(X_test.shape)
		print(y_test.shape)
		n_x, batch, input_size1, input_size2, channel = X_train.shape
		n_y, output_size = y_train.shape
		
		self.model = tf.keras.Sequential()
		self.model.add(tf.keras.Input(shape=(batch, input_size1, input_size2, channel)))
		self.model.add(tf.keras.layers.Conv2D(filters=1,kernel_size=5,input_shape=(batch, input_size1, input_size2, channel),activation='relu'))
		#self.model.add(tf.keras.layers.Conv2D(filters=1,kernel_size=5,input_shape=(batch, input_size1-4, input_size2-4, channel),activation='relu'))
		#self.model.add(tf.keras.layers.Conv2D(filters=1,kernel_size=5,input_shape=(batch, input_size1-8, input_size2-8, channel),activation='relu'))
		self.model.add(tf.keras.layers.Flatten())
		self.model.add(tf.keras.layers.Dense(4,activation='softmax'))
		self.model.compile(optimizer='adam',
		              loss='categorical_crossentropy',
		              metrics=[tf.keras.metrics.CategoricalAccuracy()])
		es_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=5)
		losses = self.model.fit(X_train,y_train,
				validation_data=(X_test,y_test),
				callbacks=[es_callback],
				epochs=1000
				)
		predictions = self.model.predict(X_test)
		y_prediction = self.model.predict(X_test)
		y_prediction = np.argmax (y_prediction, axis = 1)
		y_test = np.argmax(y_test, axis=1)
		conf_matrix = tf.math.confusion_matrix(y_test, y_prediction)
		print(conf_matrix)
		conf_matrix=conf_matrix.numpy()
		print("Acurácia Pessimo: ",(conf_matrix[0,0]/(conf_matrix[0,0]+conf_matrix[0,1]+conf_matrix[0,2]+conf_matrix[0,3])))
		print("Acurácia Ruim: ",(conf_matrix[1,1]/(conf_matrix[1,0]+conf_matrix[1,1]+conf_matrix[1,2]+conf_matrix[1,3])))
		print("Acurácia Boa: ",(conf_matrix[2,2]/(conf_matrix[2,0]+conf_matrix[2,1]+conf_matrix[2,2]+conf_matrix[2,3])))
		print("Acurácia Excelente: ",(conf_matrix[3,3]/(conf_matrix[3,0]+conf_matrix[3,1]+conf_matrix[3,2]+conf_matrix[3,3])))
		return losses

	def save_model(self,filepath):
		self.model.save(filepath)

	def load_model(self,filepath):
		self.model = tf.keras.models.load_model(filepath)

	def predict_from_opencv(self,image):
		prediction=self.model.predict(np.array([np.expand_dims(self.ANC(image),axis=(0,3))],dtype=float)/255)
		maximum=np.max(prediction)
		if prediction[0,0]==maximum: #Caso pessimo
			return 0
		elif prediction[0,1]==maximum: #Caso ruim
			return 1
		elif prediction[0,2]==maximum: #Caso bom
			return 2
		elif prediction[0,3]==maximum: #Caso excelente
			return 3
		#return self.model.predict(np.array([np.expand_dims(self.ANC(image),axis=(0,3))],dtype=float)/255)

	def predict_from_file(self,imagepath):
		image=cv2.imread(imagepath)
		prediction=self.model.predict(np.array([np.expand_dims(self.ANC(image),axis=(0,3))],dtype=float)/255)
		maximum=np.max(prediction)
		if prediction[0,0]==maximum: #Caso pessimo
			return 0
		elif prediction[0,1]==maximum: #Caso ruim
			return 1
		elif prediction[0,2]==maximum: #Caso bom
			return 2
		elif prediction[0,3]==maximum: #Caso excelente
			return 3

def validation():
	pathRoot=os.getcwd()+'\\organizacao_original\\Cam 077.3'
	tabela=np.zeros((3,3))
	for root, dirs, files in os.walk(pathRoot+'\\Verde'):
			for file in files:
				imagem=cv2.imread(os.path.join(root,file))
				if imagem is not None:
					prediction=cnn.predict(imagem)
					prediction-=1
					if prediction==-1 or prediction==0:
						prediction+=1
					tabela[int(prediction),2]+=1
					#if prediction!=3:
						#cv2.imwrite(pathRoot+'\\Erro predict\\Erro Verde\\'+str(prediction)+'\\'+file,imagem)
		
	for root, dirs, files in os.walk(pathRoot+'\\Amarela'):
			for file in files:
				imagem=cv2.imread(os.path.join(root,file))
				if imagem is not None:
					prediction=cnn.predict(imagem)
					prediction-=1
					if prediction==-1 or prediction==0:
						prediction+=1
					tabela[int(prediction),1]+=1
					#if prediction!=2:
						#cv2.imwrite(pathRoot+'\\Erro predict\\Erro Amarela\\'+str(prediction)+'\\'+file,imagem)

	for root, dirs, files in os.walk(pathRoot+'\\Vermelha'):
			for file in files:
				imagem=cv2.imread(os.path.join(root,file))
				if imagem is not None:
					prediction=cnn.predict(imagem)
					prediction-=1
					if prediction==-1 or prediction==0:
						prediction+=1
					tabela[int(prediction),0]+=1
					#if prediction!=0 or prediction!=1:
						#cv2.imwrite(pathRoot+'\\Erro predict\\Erro Vermelha\\'+str(prediction)+'\\'+file,imagem)

	tabela=np.append([['Vermelha'],['Amarela'],['Verde']],tabela,axis=-1)
	tabela=np.append([['Predição\Realidade','Vermelha','Amarela','Verde']],tabela,axis=0)

	print(tabulate(tabela))


# ------Training routine------
model=ancCNN() #Instanciate the model
model.train_model(os.getcwd()+'\\dataset_pecem_augmented') #Train the model
model.save_model(os.getcwd()+'\\ancCnn') #Save the model to file



# ------Loading routine-------
#model=ancCNN() #Instanciate the model
#model.load_model(os.getcwd()+'\\ancCnn') #Load the model from file

# -----Predict routine--------
#model.predict(imagepath)
# -----------OR---------------
#model.predict(image) #Where image is an opencv object
# Possible outputs: {0,1,2,3} that represents Pessimo, Ruim, Bom e Excelente, respectivamente.

