import cv2
import numpy as np
   
image = cv2.imread('C:\\Users\\lucas vitoriano\\Desktop\\ProjetoPecem\\Carlos-SVMClassifier\\dataset_pecem\\Ruim\\Imagem7.jpg')
#filteredImage = cv2.GaussianBlur(image, (31,31), 0)
filteredImage = cv2.Laplacian(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), cv2.CV_64F,ksize=15)
imageEx = cv2.imread('C:\\Users\\lucas vitoriano\\Desktop\\ProjetoPecem\\Carlos-SVMClassifier\\dataset_pecem\\Excelente\\Imagem1.jpg')
#filteredImage = cv2.GaussianBlur(image, (31,31), 0)
filteredImageEx = cv2.Laplacian(cv2.cvtColor(imageEx, cv2.COLOR_BGR2GRAY), cv2.CV_64F,ksize=15)
'''
cv2.imshow('Original image',image)
cv2.imshow('HSV image', hsvImage)
   
cv2.waitKey(0)
cv2.destroyAllWindows()
'''
#cv2.imshow('Canal 0',hsvImage[0])
#cv2.imshow('Canal 1',hsvImage[1])
#cv2.imshow('Excelent Image',filteredImageEx)
print("Image Excelent coeficient=",np.mean(filteredImageEx))
#cv2.imshow('Blurred Image',filteredImage)
print("Image Blurred coeficient=",np.mean(filteredImage))
cv2.waitKey(0)
cv2.destroyAllWindows()