import cv2
import numpy as np
import random
import os
import pickle
from sklearn.svm import SVC
import random

class SvmModel:
	
	def __init__(self,model_file):
		with open(model_file, 'rb') as f:
			self.clf = pickle.load(f)
	
	def classify(self,image):
		parameter=rmse(image,cv2.GaussianBlur(image,(41,41),0))
		return self.clf.predict(parameter)

path = os.getcwd()+'\\dataset_pecem_merged\\cam_112'
list_images=[]
for root, dirs, files in os.walk(path):
	for file in files:
		list_images.append(os.path.join(root,file))

image_path=random.choice(list_images)

image=cv2.imread(image_path)

svmModel = SvmModel('svmClassifierModel.pkl')
classification=svmModel.classify(image)

if classification==0:
	print("Precisa limpar")
	cv2.imshow("Precisa limpar",image)
elif classification==1:
	print("Não precisa limpar")
	cv2.imshow("Não precisa limpar",image)