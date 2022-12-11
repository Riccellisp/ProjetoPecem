# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 09:51:54 2022

@author: lueli
"""

try:
    import skvideo.io
except ImportError:
    import pip
    pip.main(['install', '--user', 'sk-video'])
    import skvideo.io
    
try:
    from numba import prange
except ImportError:
    import pip
    pip.main(['install', '--user', 'numba'])
    from numba import prange    

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import operator
import random
#import plotly.express as px
from sklearn.cluster import KMeans
import warnings 
from scipy import stats
import cv2
import os
camera_folder = 'C:\\Users\\lueli\\ProjetoPecem\\cam_50'
imgs = []
for filename in os.listdir(camera_folder):
    img = cv2.imread(os.path.join(camera_folder,filename))
    img = cv2.resize(img,(800,1600), interpolation = cv2.INTER_AREA)
    print(img.shape)
    imgs.append(img)

#img = cv2.imread('C:\\Users\\lueli\\ProjetoPecem\\imagens_nao_rotuladas\\dataset_rotulado\\Imagem1.jpg')
imgs = np.array(imgs)

print(imgs.shape)

background = imgs[0].copy()

n_frames = 8
frames_idx = []
for i in range(0, n_frames):
  frames_idx.append(random.randint(0, imgs.shape[0]-1))

for frame in frames_idx:
  
  plt.imshow(imgs[frame])
  plt.show()
  


for x in prange(0, imgs.shape[2]):
  for y in prange(0, imgs.shape[1]):
    colors=[]
    for z in frames_idx:
      colors.append(imgs[z][y][x])
    ca = KMeans(n_clusters = 2)
    ca = ca.fit(colors)
    labels, counts = np.unique(ca.labels_, return_counts=True)
    clusters = dict(zip(labels, counts))
    most_common = max(clusters, key=clusters.get)
    colors_plot = colors
    if most_common == 1:
      colors = [colors[i] for i in np.where(np.array(ca.labels_))[0]]
    else:
      colors = [colors[i] for i in np.where(np.logical_not(np.array(ca.labels_)))[0]]
    colors_a = np.array(colors)
    background_pixel = []
    background_pixel.append(np.median(colors_a[:,0]))
    background_pixel.append(np.median(colors_a[:,1]))
    background_pixel.append(np.median(colors_a[:,2]))
    background[y][x] = background_pixel  
    
breakpoint()    