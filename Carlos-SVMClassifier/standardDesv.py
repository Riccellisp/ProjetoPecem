import cv2
import numpy as np
import math

def standardDesv(img,rowSample,columnSample):
	
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
			somatory=somatory+np.std(grayImg[i*rowSample:(i+1)*rowSample,j*columnSample:(j+1)*rowSample])
	if incompleteColumn==1:
		for i in range(0,nRows):
			somatory=somatory+np.std(grayImg[i*rowSample:(i+1)*rowSample,nColumns*columnSample:columnSize])
			nBlocks = nBlocks + 1
	if incompleteRow==1:
		for j in range(0,nColumns):
			somatory=somatory+np.std(grayImg[grayImg[nRows*rowSample:rowSize,j*columnSample:(j+1)*columnSample]])
			nBlocks = nBlocks + 1
	if incompleteRow==1 and incompleteColumn==1:
		somatory=somatory+np.std(grayImg[nRows*rowSample:rowSize,nColumns*columnSample:columnSize])
		nBlocks = nBlocks + 1
	return somatory/nBlocks

def stdSpecial(img,rowSample,columnSample):
	grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	blurredImg=cv2.GaussianBlur(grayImg, (21,21), 0)
	rowSize, columnSize = grayImg.shape
	nRows = int(rowSize/rowSample)
	nColumns = int(columnSize/columnSample)
	incompleteRow = math.ceil(rowSize/rowSample) - nRows
	incompleteColumn = math.ceil(columnSize/columnSample) - nColumns

	somatory = 0
	nBlocks = nRows*nColumns

	for i in range(0,nRows):
		for j in range(0,nColumns):
			sample=grayImg[i*rowSample:(i+1)*rowSample,j*columnSample:(j+1)*rowSample]
			sampleBlurred=blurredImg[i*rowSample:(i+1)*rowSample,j*columnSample:(j+1)*rowSample]
			somatory=somatory+np.std(sample)-np.std(sampleBlurred)
	if incompleteColumn==1:
		for i in range(0,nRows):
			sample=grayImg[i*rowSample:(i+1)*rowSample,nColumns*columnSample:columnSize]
			sample=blurredImg[i*rowSample:(i+1)*rowSample,nColumns*columnSample:columnSize]
			somatory=somatory+np.std(sample)-np.std(sampleBlurred)
			nBlocks = nBlocks + 1
	if incompleteRow==1:
		for j in range(0,nColumns):
			sample=grayImg[nRows*rowSample:rowSize,j*columnSample:(j+1)*columnSample]
			sampleBlurred=blurredImg[nRows*rowSample:rowSize,j*columnSample:(j+1)*columnSample]
			somatory=somatory+np.std(sample)-np.std(sampleBlurred)
			nBlocks = nBlocks + 1
	if incompleteRow==1 and incompleteColumn==1:
		sample=grayImg[nRows*rowSample:rowSize,nColumns*columnSample:columnSize]
		sampleBlurred=blurredImg[nRows*rowSample:rowSize,nColumns*columnSample:columnSize]
		somatory=somatory+np.std(sample)-np.std(sampleBlurred)
		nBlocks = nBlocks + 1
	return somatory/nBlocks


def emeSpecial(img,rowSample,columnSample):
	grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	blurredImg=cv2.GaussianBlur(grayImg, (11,11), 10)
	rowSize, columnSize = grayImg.shape
	nRows = int(rowSize/rowSample)
	nColumns = int(columnSize/columnSample)
	incompleteRow = math.ceil(rowSize/rowSample) - nRows
	incompleteColumn = math.ceil(columnSize/columnSample) - nColumns

	somatory = 0
	nBlocks = nRows*nColumns

	for i in range(0,nRows):
		for j in range(0,nColumns):
			sample=grayImg[i*rowSample:(i+1)*rowSample,j*columnSample:(j+1)*rowSample]
			sampleBlurred=blurredImg[i*rowSample:(i+1)*rowSample,j*columnSample:(j+1)*rowSample]
			imax=[sample.max(),sampleBlurred.max()]
			imin=[sample.min(),sampleBlurred.min()]
			if imin[0]==0:
				imin[0]=1
			if imin[1]==0:
				imin[1]=1
			if imax[0]==0:
				imax[0]=1
			if imax[1]==0:
				imax[1]=1
			somatory = somatory + 20*(math.log(imax[0]/imin[0])-math.log(imax[1]/imin[1]))
	if incompleteColumn==1:
		for i in range(0,nRows):
			sample=grayImg[i*rowSample:(i+1)*rowSample,nColumns*columnSample:columnSize]
			sampleBlurred=blurredImg[i*rowSample:(i+1)*rowSample,nColumns*columnSample:columnSize]
			imax=[sample.max(),sampleBlurred.max()]
			imin=[sample.min(),sampleBlurred.min()]
			if imin[0]==0:
				imin[0]=1
			if imin[1]==0:
				imin[1]=1
			if imax[0]==0:
				imax[0]=1
			if imax[1]==0:
				imax[1]=1
			somatory = somatory + 20*(math.log(imax[0]/imin[0])-math.log(imax[1]/imin[1]))
			nBlocks = nBlocks + 1
	if incompleteRow==1:
		for j in range(0,nColumns):
			sample=grayImg[nRows*rowSample:rowSize,j*columnSample:(j+1)*columnSample]
			sampleBlurred=blurredImg[nRows*rowSample:rowSize,j*columnSample:(j+1)*columnSample]
			imax=[sample.max(),sampleBlurred.max()]
			imin=[sample.min(),sampleBlurred.min()]
			if imin[0]==0:
				imin[0]=1
			if imin[1]==0:
				imin[1]=1
			if imax[0]==0:
				imax[0]=1
			if imax[1]==0:
				imax[1]=1
			somatory = somatory + 20*(math.log(imax[0]/imin[0])-math.log(imax[1]/imin[1]))
			nBlocks = nBlocks + 1
	if incompleteRow==1 and incompleteColumn==1:
		sample=grayImg[nRows*rowSample:rowSize,nColumns*columnSample:columnSize]
		sampleBlurred=blurredImg[nRows*rowSample:rowSize,nColumns*columnSample:columnSize]
		imax=[sample.max(),sampleBlurred.max()]
		imin=[sample.min(),sampleBlurred.min()]
		if imin[0]==0:
			imin[0]=1
		if imin[1]==0:
			imin[1]=1
		if imax[0]==0:
			imax[0]=1
		if imax[1]==0:
			imax[1]=1
		somatory = somatory + 20*(math.log(imax[0]/imin[0])-math.log(imax[1]/imin[1]))
		nBlocks = nBlocks + 1
	return somatory/nBlocks

def rmse(img1,img2):
	error = np.subtract(img1,img2)
	sqrtError = np.square(error)
	meanSqrtError = np.mean(sqrtError)
	return math.sqrt(meanSqrtError)