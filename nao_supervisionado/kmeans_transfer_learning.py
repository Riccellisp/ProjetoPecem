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
from vgg16_feature_extraction import vgg16_feature_extraction
import glob
from sklearn.metrics import confusion_matrix


def image_to_pandas(image):
    df = pd.DataFrame(image.flatten()).T
    return df



def load_images_from_folder(folder):
    features = []
    filenames = []
    y_true = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            parameter=vgg16_feature_extraction(img).reshape(-1,1)
            filenames.append(filename)
            features.append(parameter)
            y_true.append()
    return features,filenames


def load_images_from_folder_2(folder):
    labels_ = os.listdir(folder)
    features = []
    labels   = []
    filenames = []
    for i, label in enumerate(labels_):
        cur_path = folder + "/" + label 
        for image_path in glob.glob(cur_path + "/*"):
            img = cv2.imread(os.path.join(folder,image_path))
            if img is not None:
                parameter=vgg16_feature_extraction(img).reshape(-1,1)
                filenames.append(image_path)
                features.append(parameter)
                labels.append(label)
    return features,filenames,labels



#df,filenames = load_images_from_folder('C:\\Users\\lueli\\ProjetoPecem\\imagens_nao_rotuladas\\all')

#df,filenames = load_images_from_folder('C:\\Users\\lueli\\ProjetoPecem\\imagens_nao_rotuladas\\dataset_rotulado')


df,filenames,labels = load_images_from_folder_2('C:\\Users\\lueli\\ProjetoPecem\\imagens_nao_rotuladas\\dataset_rotulado\\dataset_separado')



arr = np.array(df)

arr2 = arr.reshape(40,4096)
#df2 = pd.DataFrame(df)


kmeans = KMeans(n_clusters=4, random_state = 42).fit(arr2)
result = kmeans.labels_


targetNames = ['Bom', 'Excelente', 'Ruim', 'Pessimo']
targetNames = np.array(sorted(targetNames))


cm = confusion_matrix(labels, result)
title='Confusion matrix'
cmap=plt.cm.Blues
normalize = True
import itertools
if normalize:
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print("Normalized confusion matrix")
else:
    print('Confusion matrix, without normalization')
tick_marks = [0,1,2,3,4,5,6,7,8]
plt.imshow(cm, interpolation='nearest', cmap=cmap)
plt.title(title)
plt.colorbar()
# plt.xticks(tick_marks, np.array(targetNames), rotation=45)
plt.yticks(tick_marks, np.array(targetNames))

fmt = '.2f' if normalize else 'd'
thresh = cm.max() / 2.
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, format(cm[i, j], fmt),
             horizontalalignment="center",
             color="white" if cm[i, j] > thresh else "black")

plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.tight_layout()




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