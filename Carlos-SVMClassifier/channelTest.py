import cv2
   
image = cv2.imread('C:\\Users\\carlo\\Documents\\GitHub\\ProjetoPecem\\Carlos-SVMClassifier\\dataset_pecem\\Ruim\\Imagem7.jpg')
blurredImage = cv2.GaussianBlur(image, (31,31), 0)
'''
cv2.imshow('Original image',image)
cv2.imshow('HSV image', hsvImage)
   
cv2.waitKey(0)
cv2.destroyAllWindows()
'''
#cv2.imshow('Canal 0',hsvImage[0])
#cv2.imshow('Canal 1',hsvImage[1])
cv2.imshow('Original Image',image)
cv2.imshow('Blurred Image',blurredImage)
cv2.waitKey(0)
cv2.destroyAllWindows()