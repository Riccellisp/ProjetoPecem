import cv2
import numpy
import math

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
			imax=1
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
			imax=1
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
			imax=1
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
		imax=1
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
