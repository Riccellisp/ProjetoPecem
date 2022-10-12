import cv2
from metrics import mse,ssim

import numpy as np
from metrics import AMBE

# AMBE
imageA = cv2.imread('images/Fig0310(b)(washed_out_pollen_image).tif',0)
imageB = cv2.equalizeHist(imageA)

ambe = AMBE(imageA, imageB)
print(f'AMBE is: {ambe}')

horizontal_concat = np.concatenate((imageA, imageB), axis=1)
cv2.imshow("Horizontal plot", horizontal_concat)
cv2.waitKey(0)
cv2.destroyAllWindows()