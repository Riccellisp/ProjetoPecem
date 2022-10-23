import cv2
import numpy
import math

def std(test_list):
	if len(test_list)==0:
		return 0
	mean = sum(test_list) / len(test_list)
	variance = sum([((x - mean) ** 2) for x in test_list]) / len(test_list)
	return variance ** 0.5

def standardDesv(img,rowSample,columnSample):
	
	grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	
	rowSize, columnSize = grayImg.shape
	nRows = int(rowSize/rowSample)
	nColumns = int(columnSize/columnSample)
	incompleteRow = math.ceil(rowSize/rowSample) - nRows
	incompleteColumn = math.ceil(columnSize/columnSample) - nColumns

	somatory = 0
	nBlocks = nRows*nColumns
	sampleList=[]
	for i in range(0,nRows):
		for j in range(0,nColumns):
			sampleList.clear()
			for x in range(i*rowSample,(i+1)*rowSample):
				for y in range(j*columnSample,(j+1)*columnSample):
					sampleList.append(grayImg[x,y])
			somatory=somatory+std(sampleList)

	if incompleteColumn==1:
		for i in range(0,nRows):
			sampleList.clear()
			for x in range(i*rowSample,(i+1)*rowSample):
				for y in range(nColumns*columnSample,columnSize):
					sampleList.append(grayImg[x,y])
			somatory=somatory+std(sampleList)
			nBlocks = nBlocks + 1
	if incompleteRow==1:
		for j in range(0,nColumns):
			sampleList.clear()
			for x in range(nRows*rowSample,rowSize):
				for y in range(j*columnSample,(j+1)*columnSample):
					sampleList.append(grayImg[x,y])
			somatory=somatory+std(sampleList)
			nBlocks = nBlocks + 1
	if incompleteRow==1 and incompleteColumn==1:
		sampleList.clear()
		for x in range(nRows*rowSample,rowSize):
			for y in range(nColumns*columnSize,columnSize):
				sampleList.append(grayImg[x,y])
		somatory=somatory+std(sampleList)
		nBlocks = nBlocks + 1
	return somatory/nBlocks
