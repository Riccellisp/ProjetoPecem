from skimage.metrics import structural_similarity as ssim
from skimage import color,filters
import matplotlib.pyplot as plt
import numpy as np
import cv2
import math
import sys

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

def eme(img,rowSample,columnSample):
	
	grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	rowSize, columnSize = grayImg.shape
	nRows = int(rowSize/rowSample)
	nColumns = int(columnSize/columnSample)
	incompleteRow = math.ceil(rowSize/rowSample) - nRows
	incompleteColumn = math.ceil(columnSize/columnSample) - nColumns

	somatory = 0
	nBlocks = nRows*nColumns
	for i in range(0,nRows):
		for j in range(0,nColumns):
			imax=0
			imin=255
			for x in range(i*rowSample,(i+1)*rowSample):
				for y in range(j*columnSample,(j+1)*columnSample):
					if grayImg[x,y]>imax:
						imax=grayImg[x,y]
					if grayImg[x,y]<imin:
						imin=grayImg[x,y]
			if imin==0:
				somatory = somatory + 20*math.log(imax/0.5)
			else:
				somatory = somatory + 20*math.log(imax/imin)

	if incompleteColumn==1:
		for i in range(0,nRows):
			imax=0
			imin=255
			for x in range(i*rowSample,(i+1)*rowSample):
				for y in range(nColumns*columnSample,columnSize):
					if grayImg[x,y]>imax:
						imax=grayImg[x,y]
					if grayImg[x,y]<imin:
						imin=grayImg[x,y]
			if imin==0:
				somatory = somatory + 20*math.log(imax/0.5)
			else:
				somatory = somatory + 20*math.log(imax/imin)
			nBlocks = nBlocks + 1
	if incompleteRow==1:
		for j in range(0,nColumns):
			imax=0
			imin=255
			for x in range(nRows*rowSample,rowSize):
				for y in range(j*columnSample,(j+1)*columnSample):
					if grayImg[x,y]>imax:
						imax=grayImg[x,y]
					if grayImg[x,y]<imin:
						imin=grayImg[x,y]
			if imin==0:
				somatory = somatory + 20*math.log(imax/0.5)
			else:
				somatory = somatory + 20*math.log(imax/imin)
			nBlocks = nBlocks + 1
	if incompleteRow==1 and incompleteColumn==1:
		imax=0
		imin=255
		for x in range(nRows*rowSample,rowSize):
			for y in range(nColumns*columnSize,columnSize):
				if grayImg[x,y]>imax:
					imax=grayImg[x,y]
				if grayImg[x,y]<imin:
					imin=grayImg[x,y]
		if imin==0:
			somatory = somatory + 20*math.log(imax/0.5)
		else:
			somatory = somatory + 20*math.log(imax/imin)
		nBlocks = nBlocks + 1
	return somatory/nBlocks

def rmse(img1,img2):
	error = np.subtract(img1,img2)
	sqrtError = np.square(error)
	meanSqrtError = np.mean(sqrtError)
	return math.sqrt(meanSqrtError)

def emee(img,rowSample,columnSample):
	
	grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	
	rowSize, columnSize = grayImg.shape
	nRows = int(rowSize/rowSample)
	nColumns = int(columnSize/columnSample)
	incompleteRow = math.ceil(rowSize/rowSample) - nRows
	incompleteColumn = math.ceil(columnSize/columnSample) - nColumns

	somatory = 0
	nBlocks = nRows*nColumns
	for i in range(0,nRows):
		for j in range(0,nColumns):
			imax=0
			imin=255
			for x in range(i*rowSample,(i+1)*rowSample):
				for y in range(j*columnSample,(j+1)*columnSample):
					if grayImg[x,y]>imax:
						imax=grayImg[x,y]
					if grayImg[x,y]<imin:
						imin=grayImg[x,y]
			if imin==0:
				somatory = somatory + imax/0.5*math.log(imax/0.5)
			else:
				somatory = somatory + imax/imin*math.log(imax/imin)

	if incompleteColumn==1:
		for i in range(0,nRows):
			imax=0
			imin=255
			for x in range(i*rowSample,(i+1)*rowSample):
				for y in range(nColumns*columnSample,columnSize):
					if grayImg[x,y]>imax:
						imax=grayImg[x,y]
					if grayImg[x,y]<imin:
						imin=grayImg[x,y]
			if imin==0:
				somatory = somatory + imax/0.5*math.log(imax/0.5)
			else:
				somatory = somatory + imax/imin*math.log(imax/imin)
			nBlocks = nBlocks + 1
	if incompleteRow==1:
		for j in range(0,nColumns):
			imax=0
			imin=255
			for x in range(nRows*rowSample,rowSize):
				for y in range(j*columnSample,(j+1)*columnSample):
					if grayImg[x,y]>imax:
						imax=grayImg[x,y]
					if grayImg[x,y]<imin:
						imin=grayImg[x,y]
			if imin==0:
				somatory = somatory + imax/0.5*math.log(imax/0.5)
			else:
				somatory = somatory + imax/imin*math.log(imax/imin)
			nBlocks = nBlocks + 1
	if incompleteRow==1 and incompleteColumn==1:
		imax=0
		imin=255
		for x in range(nRows*rowSample,rowSize):
			for y in range(nColumns*columnSize,columnSize):
				if grayImg[x,y]>imax:
					imax=grayImg[x,y]
				if grayImg[x,y]<imin:
					imin=grayImg[x,y]
		if imin==0:
			somatory = somatory + imax/0.5*math.log(imax/0.5)
		else:
			somatory = somatory + imax/imin*math.log(imax/imin)
		nBlocks = nBlocks + 1
	return somatory/nBlocks

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

def variance_of_laplacian(image):
	'''
	compute the Laplacian of the image and then return the focus
	measure, which is simply the variance of the Laplacian
	''' 
	return cv2.Laplacian(image, cv2.CV_64F).var()


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

def plipsum(i,j,gamma=1026):
    return i + j - i * j / gamma

def plipsub(i,j,k=1026):
    return k * (i - j) / (k - j)

def plipmult(c,j,gamma=1026):
    return gamma - gamma * (1 - j / gamma)**c

def logamee(ch,blocksize=8):

    num_x = math.ceil(ch.shape[0] / blocksize)
    num_y = math.ceil(ch.shape[1] / blocksize)
    
    s = 0
    w = 1. / (num_x * num_y)
    for i in range(num_x):

        xlb = i * blocksize
        if i < num_x - 1:
            xrb = (i+1) * blocksize
        else:
            xrb = ch.shape[0]

        for j in range(num_y):

            ylb = j * blocksize
            if j < num_y - 1:
                yrb = (j+1) * blocksize
            else:
                yrb = ch.shape[1]
            
            block = ch[xlb:xrb,ylb:yrb]
            blockmin = np.float(np.min(block))
            blockmax = np.float(np.max(block))

            top = plipsub(blockmax,blockmin)
            bottom = plipsum(blockmax,blockmin)

            m = top/bottom
            if m ==0.:
                s+=0
            else:
                s += (m) * np.log(m)

    return plipmult(w,s)

def UIQM(a,p1=0.0282,p2=0.2953,p3=3.5753):
	"""
	REF: K. Panetta, C. Gao and S. Agaian, Human-Visual-System-Inspired Underwater Image Quality Measures, 
	in IEEE Journal of Oceanic Engineering, vol. 41, no. 3, pp. 541-551, July 2016, doi: 10.1109/JOE.2015.2469915.

	Metrica sem referencia, semelhante a UCIQE, mas mais atual. Leva em consideração a medida de colorção,
	medida de nitidez e medida de contraste.
	"""
	#1st term UICM
	#TαL=⌈αLK⌉ - > o inteiro mais próximo maior ou igual a αLK
	#TαR=⌊αRK⌋ - > o inteiro mais próximo menor ou igual a αRK
	rgb=a
	gray = color.rgb2gray(a)
	rg = rgb[:,:,0] - rgb[:,:,1]
	yb = (rgb[:,:,0] + rgb[:,:,1]) / 2 - rgb[:,:,2]
	rgl = np.sort(rg,axis=None)
	ybl = np.sort(yb,axis=None)
	al1 = 0.1
	al2 = 0.1
	T1 = np.int(al1 * len(rgl))
	T2 = np.int(al2 * len(rgl))
	rgl_tr = rgl[T1:-T2]
	ybl_tr = ybl[T1:-T2]

	urg = np.mean(rgl_tr) # μ^2_α,RG
	s2rg = np.mean((rgl_tr - urg) ** 2)  # σ2α,RG
	uyb = np.mean(ybl_tr) # μ^2_α,YB
	s2yb = np.mean((ybl_tr- uyb) ** 2) # σ2α,YB

	uicm =-0.0268 * np.sqrt(urg**2 + uyb**2) + 0.1586 * np.sqrt(s2rg + s2yb)

	#2nd term UISM (k1k2=8x8)   # medida de nitidez de imagem
	"""Para medir a nitidez nas bordas, o detector de bordas Sobel é aplicado primeiro em cada componente de cor RGB
	   O mapa de arestas resultante é então multiplicado pela imagem original para obter o mapa de arestas em tons de cinza."""
	# Rsobel = rgb[:,:,0] * filters.sobel(rgb[:,:,0]) 
	# Gsobel = rgb[:,:,1] * filters.sobel(rgb[:,:,1])
	# Bsobel = rgb[:,:,2] * filters.sobel(rgb[:,:,2])

	# Rsobel=np.round(Rsobel).astype(np.uint8)  # Arredonda a matriz para numeros inteiros
	# Gsobel=np.round(Gsobel).astype(np.uint8)
	# Bsobel=np.round(Bsobel).astype(np.uint8)

	Reme = eme(a,8,8)
	Geme = eme(a,8,8)
	Beme = eme(a,8,8)
	uism = 0.299 * Reme + 0.587 * Geme + 0.114 * Beme

	#3rd term UIConM
	uiconm = logamee(gray)
	uiqm = p1 * uicm + p2 * uism + p3 * uiconm

	return uiqm	

def CCF(imageRGB):
	"""REF:
	@article{WANG2018904,
	title = {An imaging-inspired no-reference underwater color image quality assessment metric},
	journal = {Computers & Electrical Engineering},
	volume = {70},
	pages = {904-913},
	year = {2018},
	issn = {0045-7906},
	doi = {https://doi.org/10.1016/j.compeleceng.2017.12.006},
	url = {https://www.sciencedirect.com/science/article/pii/S0045790617324953},
	author = {Yan Wang and Na Li and Zongying Li and Zhaorui Gu and Haiyong Zheng and Bing Zheng and Mengnan Sun},
	keywords = {No-reference image quality assessment, Underwater imaging, Underwater image, Underwater color image quality},
	abstract = {Underwater color image quality assessment (IQA) plays an important role in analysis and applications of underwater imaging as well as image processing algorithms. This paper presents a new metric inspired by the imaging analysis on underwater absorption and scattering characteristics, dubbed the CCF. This metric is feature-weighted with a combination of colorfulness index, contrast index and fog density index, which can quantify the color loss caused by absorption, the blurring caused by forward scattering and the foggy caused by backward scattering, respectively. Then multiple linear regression is used to calculate three weighted coefficients. A new underwater image database is built to illustrate the performance of the proposed metric. Experimental results show a strong correlation between the proposed metric and mean opinion score (MOS). The proposed CCF metric outperforms many of the leading atmospheric IQA metrics, and it can effectively assess the performance of underwater image enhancement and image restoration methods.}
	}
	
	Métrica sem referência. Essa métrica é ponderada por recursos com uma combinação de índice de colorido, 
	índice de contraste e índice de densidade de neblina, que pode quantificar a perda de cor 
	causada pela absorção, o desfoque causado pela dispersão para frente e o nevoeiro causado 
	pela dispersão para trás, respectivamente.
	"""
	rgb=imageRGB
	gray = color.rgb2gray(imageRGB)
	# Passo 1
	Rij=np.log(rgb[:,:,0])-np.mean(rgb[:,:,0])
	Gij=np.log(rgb[:,:,1])-np.mean(rgb[:,:,1])	
	Bij=np.log(rgb[:,:,2])-np.mean(rgb[:,:,2])

	# Passo 2
	a = rgb[:,:,0] - rgb[:,:,1]
	b = (rgb[:,:,0] + rgb[:,:,1]) / 2 - rgb[:,:,2]
	
	var_a  = np.var(a)
	var_b  = np.var(b)
	mean_a = np.mean(a)
	mean_b = np.mean(b)

	ccf=(math.sqrt(var_a+var_b)+0.3*math.sqrt(var_a+var_b)) /85.59
	

	return ccf