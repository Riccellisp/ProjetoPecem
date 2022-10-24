import cv2
   
image = cv2.imread('C:\\Users\\carlo\\Documents\\GitHub\\ProjetoPecem\\Carlos-SVMClassifier\\dataset_pecem\\Ruim\\Imagem7.jpg')
hsvImage = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
'''
cv2.imshow('Original image',image)
cv2.imshow('HSV image', hsvImage)
   
cv2.waitKey(0)
cv2.destroyAllWindows()
'''
#cv2.imshow('Canal 0',hsvImage[0])
#cv2.imshow('Canal 1',hsvImage[1])
cv2.imshow('Canal 2',hsvImage[:,:,1])
cv2.waitKey(0)
cv2.destroyAllWindows()