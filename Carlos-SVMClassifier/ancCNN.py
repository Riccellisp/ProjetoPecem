import os
import tensorflow as tf
import cv2
import numpy as np
import math
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
		return responseMatrix

	def generate_dataset(self,pathDatabase):

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
				parameter=np.expand_dims(self.ANC(img),axis=(0,3))
				parameters_excelente.append(parameter)
			else:
				print(path)

		for path in list_bom:
			img=cv2.imread(path)
			if img is not None:
				parameter=np.expand_dims(self.ANC(img),axis=(0,3))
				parameters_bom.append(parameter)
			else:
				print(path)

		for path in list_ruim:
			img=cv2.imread(path)
			if img is not None:
				parameter=np.expand_dims(self.ANC(img),axis=(0,3))
				parameters_ruim.append(parameter)
			else:
				print(path)

		for path in list_pessimo:
			img=cv2.imread(path)
			if img is not None:
				parameter=np.expand_dims(self.ANC(img),axis=(0,3))
				parameters_pessimo.append(parameter)
			else:
				print(path)

		X = np.array(parameters_excelente+parameters_bom+parameters_ruim+parameters_pessimo,dtype=float)/255
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
		dataset=[X,Y]

		return dataset

	def train_model(self,pathDatabase):
		dataset=self.generate_dataset(pathDatabase)
		n_x, batch, input_size1, input_size2, channel = dataset[0].shape
		n_y, output_size = dataset[1].shape
		X_train, X_test, y_train, y_test = train_test_split(dataset[0], dataset[1])

		self.model = tf.keras.Sequential()
		self.model.add(tf.keras.Input(shape=(batch, input_size1, input_size2, channel)))
		self.model.add(tf.keras.layers.Conv2D(filters=1,kernel_size=3,input_shape=(batch, input_size1, input_size2, channel),activation='relu'))
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

