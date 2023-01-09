import cv2,random,os
import numpy as np

def changeBrightness(img,low,high):
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

def horizontalFlip(img):
	return cv2.flip(img,int(random.getrandbits(1)))

def verticalFlip(img):
	return cv2.flip(img,int(random.getrandbits(1)))

def rotation(img,max_angle):
    angle = int(random.uniform(-max_angle, max_angle))
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((int(w/2), int(h/2)), angle, 1)
    img = cv2.warpAffine(img, M, (w, h))
    return img

pathDatabase=os.getcwd()+'\\dataset_pecem'

pathExcelente=pathDatabase+'\\Excelente'
pathBom=pathDatabase+'\\Bom'
pathRuim=pathDatabase+'\\Ruim'
pathPessimo=pathDatabase+'\\Pessimo'

pathAugmented=os.getcwd()+'\\dataset_pecem_augmented'

list_excelente=[]
list_bom=[]
list_ruim=[]
list_pessimo=[]

count=1


for root, dirs, files in os.walk(pathExcelente):
    for file in files:
        #list_excelente.append(os.path.join(root,file))
        imagem=cv2.imread(os.path.join(root,file))
        if imagem is not None:
            for i in range(0,20):
                image=changeBrightness(imagem,0.8,1.2)
                image=horizontalFlip(image)
                image=verticalFlip(image)
                image=rotation(image,30)
                cv2.imwrite(pathAugmented+'\\Excelente\\Imagem'+str(count)+'.'+str(i)+'.jpg',image)
            count=count+1

for root, dirs, files in os.walk(pathBom):
    for file in files:
        #list_bom.append(os.path.join(root,file))
        imagem=cv2.imread(os.path.join(root,file))
        if imagem is not None:
            for i in range(0,20):
                image=changeBrightness(imagem,0.8,1.2)
                image=horizontalFlip(image)
                image=verticalFlip(image)
                image=rotation(image,30)
                cv2.imwrite(pathAugmented+'\\Bom\\Imagem'+str(count)+'.'+str(i)+'.jpg',image)
            count=count+1

for root, dirs, files in os.walk(pathRuim):
    for file in files:
        #list_ruim.append(os.path.join(root,file))
        imagem=cv2.imread(os.path.join(root,file))
        if imagem is not None:
            for i in range(0,20):
                image=changeBrightness(imagem,0.8,1.2)
                image=horizontalFlip(image)
                image=verticalFlip(image)
                image=rotation(image,30)
                cv2.imwrite(pathAugmented+'\\Ruim\\Imagem'+str(count)+'.'+str(i)+'.jpg',image)
            count=count+1

for root, dirs, files in os.walk(pathPessimo):
    for file in files:
        #list_pessimo.append(os.path.join(root,file))
        imagem=cv2.imread(os.path.join(root,file))
        if imagem is not None:
            for i in range(0,20):
                image=changeBrightness(imagem,0.8,1.2)
                image=horizontalFlip(image)
                image=verticalFlip(image)
                image=rotation(image,30)
                cv2.imwrite(pathAugmented+'\\Pessimo\\Imagem'+str(count)+'.'+str(i)+'.jpg',image)
            count=count+1