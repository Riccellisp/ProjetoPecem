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
	hsvImg = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
	rowSize, columnSize, channel = hsvImg.shape
	nRows = int(rowSize/rowSample)
	nColumns = int(columnSize/columnSample)
	incompleteRow = math.ceil(rowSize/rowSample) - nRows
	incompleteColumn = math.ceil(columnSize/columnSample) - nColumns

	somatory = 0
	nBlocks = nRows*nColumns

	for i in range(0,nRows):
		for j in range(0,nColumns):
			somatory=somatory+np.std(hsvImg[i*rowSample:(i+1)*rowSample,j*columnSample:(j+1)*rowSample,1])
	if incompleteColumn==1:
		for i in range(0,nRows):
			somatory=somatory+np.std(hsvImg[i*rowSample:(i+1)*rowSample,nColumns*columnSample:columnSize,1])
			nBlocks = nBlocks + 1
	if incompleteRow==1:
		for j in range(0,nColumns):
			somatory=somatory+np.std(hsvImg[nRows*rowSample:rowSize,j*columnSample:(j+1)*columnSample,1])
			nBlocks = nBlocks + 1
	if incompleteRow==1 and incompleteColumn==1:
		somatory=somatory+np.std(hsvImg[nRows*rowSample:rowSize,nColumns*columnSample:columnSize,1])
		nBlocks = nBlocks + 1
	return somatory/nBlocks
