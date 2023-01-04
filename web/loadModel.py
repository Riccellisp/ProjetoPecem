import cv2
import numpy as np
import random
import os
import math
import pickle
from sklearn.svm import SVC
import random


#######Classe do modelo##########################################################
class SvmModel:
	
	def __init__(self,model_file):#Carrega o modelo do arquivo
		with open(model_file, 'rb') as f:
			self.clf = pickle.load(f)
	
	def classify(self,image):#Classifica uma imagem (formate opencv) em Precisa limpar(0) ou Não precisa limpar(1)
		error = np.subtract(image,cv2.GaussianBlur(image,(41,41),0))
		sqrtError = np.square(error)
		meanSqrtError = np.mean(sqrtError)
		parameter = math.sqrt(meanSqrtError)
		return self.clf.predict([[0,parameter]])
##################################################################################

#Registro das imagens para teste####################
path = os.getcwd()+'\\dataset_pecem_merged\\cam_321'
list_images=[]
for root, dirs, files in os.walk(path):
	for file in files:
		list_images.append(os.path.join(root,file))
####################################################

#Escolha de uma imagem aleatória da lista#
image_path=random.choice(list_images)

image=cv2.imread(image_path)
##########################################

#Carregar o modelo e classificar a imagem#####
svmModel = SvmModel('svmClassifierModel.pkl')
classification=svmModel.classify(image)
##############################################

#Exibe a imagem e se a câmera deve ser limpa ou não#
if classification==0:
	print("Precisa limpar")
	cv2.imshow("Precisa limpar",image)
elif classification==1:
	print("Não precisa limpar")
	cv2.imshow("Não precisa limpar",image)

cv2.waitKey(0)
cv2.destroyAllWindows()
####################################################