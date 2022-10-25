import cv2
import numpy as np
import os
from standardDesv import rmse

pathDatabase=os.getcwd()+'\\dataset_pecem'
pathExcelente=pathDatabase+'\\Excelente'
pathBom=pathDatabase+'\\Bom'
pathRuim=pathDatabase+'\\Ruim'
pathPessimo=pathDatabase+'\\Pessimo'

tabela=[]
dtype=[('path',np.unicode_, 200),('metric',float)]
for root, dirs, files in os.walk(pathExcelente):
	for file in files:
		path=os.path.join(root,file)
		img=cv2.imread(path)
		tabela.append((path,rmse(img,cv2.GaussianBlur(img,(31,31),0))))

for root, dirs, files in os.walk(pathBom):
	for file in files:
		path=os.path.join(root,file)
		img=cv2.imread(path)
		tabela.append((path,rmse(img,cv2.GaussianBlur(img,(31,31),0))))

for root, dirs, files in os.walk(pathRuim):
	for file in files:
		path=os.path.join(root,file)
		img=cv2.imread(path)
		tabela.append((path,rmse(img,cv2.GaussianBlur(img,(31,31),0))))

for root, dirs, files in os.walk(pathPessimo):
	for file in files:
		path=os.path.join(root,file)
		img=cv2.imread(path)
		tabela.append((path,rmse(img,cv2.GaussianBlur(img,(31,31),0))))

matriz=np.array(tabela,dtype=dtype)
print(matriz)
sort_matriz=np.sort(matriz,order='metric')
count=1
print(sort_matriz)
for element in sort_matriz:
	path,value=element
	#print(element)
	img=cv2.imread(path)
	cv2.imwrite('Noise rank '+str(count)+'.jpg',img)
	count=count+1