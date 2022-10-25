import cv2
import numpy as np
import os
from standardDesv import rmse

pathDatabase=os.getcwd()+'\\dataset_pecem'
pathExcelente=pathDatabase+'\\Excelente'
pathBom=pathDatabase+'\\Bom'
pathRuim=pathDatabase+'\\Ruim'
pathPessimo=pathDatabase+'\\Pessimo'

list_files=[]
parameters=[]

for root, dirs, files in os.walk(pathExcelente):
	for file in files:
		list_files.append(os.path.join(root,file))

for root, dirs, files in os.walk(pathBom):
	for file in files:
		list_files.append(os.path.join(root,file))

for root, dirs, files in os.walk(pathRuim):
	for file in files:
		list_files.append(os.path.join(root,file))

for root, dirs, files in os.walk(pathPessimo):
	for file in files:
		list_files.append(os.path.join(root,file))

for i in range(0,len(list_files)):
	img=cv2.imread(list_files[i])
	parameters.append(rmse(img,cv2.GaussianBlur(img,(31,31),0)))


tabela=np.array((list_files,parameters))
sort_tabela=np.sort(tabela,axis=-1)
count=1
print(sort_tabela[0][:])
for path in sort_tabela[0][:]:
	img=cv2.imread(path)
	cv2.imwrite('\\dataset_sorted\\Noise rank '+str(count)+'.jpg',img)
	count=count+1