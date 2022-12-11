try:
    from rembg import remove
except ImportError:
    import pip
    pip.main(['install', '--user', 'rembg'])
    from rembg import remove

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import operator
import random
#from numba import prange
#import plotly.express as px
from sklearn.cluster import KMeans
import warnings 
from scipy import stats
import cv2
import os
camera_folder = 'C:\\Users\\lueli\\ProjetoPecem\\cam_50'
imgs = []

#for filename in os.listdir(camera_folder):
#    img = cv2.imread(os.path.join(camera_folder,filename))
#    print(img.shape)
#    imgs.append(img)

#img = cv2.imread('C:\\Users\\lueli\\ProjetoPecem\\imagens_nao_rotuladas\\dataset_rotulado\\Imagem1.jpg')
#imgs = np.array(imgs)
#print(imgs.shape)


input_path = 'C:\\Users\\lueli\\ProjetoPecem\\imagens_nao_rotuladas\\dataset_rotulado\\Imagem5.jpg'
output_path = 'C:\\Users\\lueli\\ProjetoPecem\\imagens_nao_rotuladas\\dataset_rotulado\\Imagem5_boa_output.png'

folder = 'C:\\Users\\lueli\\ProjetoPecem\\imagens_nao_rotuladas\\bgremoval'

for filename in os.listdir(folder):
    img = cv2.imread(os.path.join(folder,filename))
    output = remove(img)
    cv2.imwrite(folder+'\\'+filename.replace('.jpg','bg-removal.png'),output)