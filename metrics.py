from skimage.metrics import structural_similarity as ssim
from skimage import color
import matplotlib.pyplot as plt
import numpy as np
import cv2

def mse(imageA, imageB):
	# the 'Mean Squared Error' between the two images is the
	# sum of the squared difference between the two images;
	# NOTE: the two images must have the same dimension
	err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
	err /= float(imageA.shape[0] * imageA.shape[1])
	
	# return the MSE, the lower the error, the more "similar"
	# the two images are
	return err

def ssim (imageA, imageB):
  return ssim(imageA, imageB)

def AMBE(imageA, imageB):
	'''
	AMBE is the abbreviation of Absolute Mean Brightness Error. It is the absolute 
	difference between the mean of input and output image. It is formally defined by:

							AMBE = | E(X) B E(Y) |

	where, X and Y denote the input and output image, respectively and E(.) denotes
	the expected value, i.e., the statistical mean. 
	Equation 1 clearly shows that AMBE is designed to detect one of the distortions-
	excessive brightness change. AMBE was proposed by Chen and Ramli (2003b) to evaluate 
	the performance in preserving original brightness.

	imageA is the raw image (before histogram equalization per example), and imageB 
	is the image after preprocessing
	'''

	return np.abs(np.mean(imageA) - np.mean(imageB))

def IEM_filter(image):
	image = image.astype(np.float32)

	(iH, iW) = image.shape[:2]
	output = np.zeros((iH-2, iW-2), dtype="float32")

	for y in np.arange(0, iH-2):
		for x in np.arange(0, iW-2):

			roi = image[y:y + 3, x:x + 3]

			k = roi
			k = (np.abs(k[1][1] - k[0][0]) + np.abs(k[1][1] - k[0][1]) + np.abs(k[1][1] - k[0][2]) +
				np.abs(k[1][1] - k[1][0]) + np.abs(k[1][1] - k[1][2]) +
				np.abs(k[1][1] - k[2][0]) + np.abs(k[1][1] - k[2][1]) + np.abs(k[1][1] - k[2][2]))  

			output[y, x] = k
	return np.sum(output)

def IEM(imageA, imageB):
	'''
	Image Enhancement Metric(IEM) approximates the contrast and
	sharpness of an image by dividing an image into non-overlapping
	blocks. 

	imageA is the raw image (before histogram equalization per example), and imageB 
	is the image after preprocessing
	'''
	valA = IEM_filter(imageA)
	valB = IEM_filter(imageB)

	return valB/valA

def UCIQE(a,c1=0.4680,c2 = 0.2745,c3 = 0.2576):
    """
    Underwater colour image quality evaluation metric (UCIQE) é uma métrica baseada na combinação
    linear de croma (pureza), saturação e contraste principalmente de imagens subaquáticas, mas também
    baseadaem trabalhos atuais de avaliação de imagens coloridas atmosféricas. 
    REF: M. Yang and A. Sowmya, "An Underwater Color Image Quality Evaluation Metric," in IEEE Transactions on Image Processing, 
    vol. 24, no. 12, pp. 6062-6071, Dec. 2015, doi: 10.1109/TIP.2015.2491020.
    
    
    :param a: imagem de entrada
    :c1,c2,c3: coeficentes ponderados
    :return c1 * sc + c2 * conl + c3 * us
    """
    rgb = a
    lab = color.rgb2lab(a)
    gray = color.rgb2gray(a)
    l = lab[:,:,0]

    #1st term
    chroma = (lab[:,:,1]**2 + lab[:,:,2]**2)**0.5
    uc = np.mean(chroma)
    sc = (np.mean((chroma - uc)**2))**0.5

    #2nd term
    top = np.int(np.round(0.01*l.shape[0]*l.shape[1]))
    sl = np.sort(l,axis=None)
    isl = sl[::-1]
    conl = np.mean(isl[:top])-np.mean(sl[:top])

    #3rd term
    satur = []
    chroma1 = chroma.flatten()
    l1 = l.flatten()
    for i in range(len(l1)):
        if chroma1[i] == 0: satur.append(0)
        elif l1[i] == 0: satur.append(0)
        else: satur.append(chroma1[i] / l1[i])

    us = np.mean(satur)

    return  c1 * sc + c2 * conl + c3 * us