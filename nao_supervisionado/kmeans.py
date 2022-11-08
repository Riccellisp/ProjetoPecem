# -*- coding: utf-8 -*-
"""
Created on Sat Nov  5 11:29:29 2022

@author: lueli
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import colors
from skimage.color import rgb2gray, rgb2hsv, hsv2rgb
from skimage.io import imread, imshow
from sklearn.cluster import KMeans
import cv2
import os

def image_to_pandas(image):
    df = pd.DataFrame(image.flatten()).T
    return df



def load_images_from_folder(folder):
    images = []
    filenames = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            filenames.append(filename)
            resized1 = cv2.resize(img,(800,1600), interpolation = cv2.INTER_AREA)
            scale_percent = 60 # percent of original size
            width = int(resized1.shape[1] * scale_percent / 100)
            height = int(resized1.shape[0] * scale_percent / 100)
            dim = (width, height)
            resized2 = cv2.resize(resized1, dim, interpolation = cv2.INTER_AREA)
            images.append(resized2.flatten())
    return images,filenames



#df,filenames = load_images_from_folder('C:\\Users\\lueli\\ProjetoPecem\\imagens_nao_rotuladas\\all')

df,filenames = load_images_from_folder('C:\\Users\\lueli\\ProjetoPecem\\imagens_nao_rotuladas\\dataset_rotulado')

df2 = pd.DataFrame(df)


kmeans = KMeans(n_clusters=4, random_state = 42).fit(df2)
result = kmeans.labels_


# open file in write mode
with open(r'C:\\Users\\lueli\\ProjetoPecem\\kmeans_labes.txt', 'w') as fp:
    for item in result:
        # write each item on a new line
        fp.write("%s\n" % item)
    print('Done')
    


# open file in write mode
with open(r'C:\\Users\\lueli\\ProjetoPecem\\filenames.txt', 'w') as fp:
    for item in filenames:
        # write each item on a new line
        fp.write("%s\n" % item)
    print('Done')    