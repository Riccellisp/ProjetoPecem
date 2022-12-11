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
try:
    from rembg import remove
except ImportError:
    import pip
    pip.main(['install', '--user', 'rembg'])
    from rembg import remove



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


def load_images_from_folder_and_fourrier(folder):
    labels_ = os.listdir(folder)
    features = []
    labels   = []
    filenames = []
    for i, label in enumerate(labels_):
        cur_path = folder + "/" + label 
        for image_path in glob.glob(cur_path + "/*"):
            img = cv2.imread(os.path.join(folder,image_path))
            if img is not None:
                img = cv2.resize(img,(224,224), interpolation = cv2.INTER_AREA)
                f = np.fft.fft2(img) # transformação de frequência
                f_abs = abs(f)
                f_norm =  f_abs/(f_abs.max()/255.0) 
                fshift = np.fft.fftshift(f_norm) # componente de frequência no centro
                spectrum = np.log(1+np.abs(fshift)).reshape(-1,1) # espectro de magnitude
                filenames.append(image_path)
                features.append(spectrum)
                labels.append(label)
    return features,filenames,labels

def load_images_from_folder_and_fourrier_and_transfer(folder):
    labels_ = os.listdir(folder)
    features = []
    labels   = []
    filenames = []
    for i, label in enumerate(labels_):
        cur_path = folder + "/" + label 
        for image_path in glob.glob(cur_path + "/*"):
            img = cv2.imread(os.path.join(folder,image_path))
            if img is not None:
                img = cv2.resize(img,(224,224), interpolation = cv2.INTER_AREA)
                f = np.fft.fft2(img) # transformação de frequência
                f_abs = abs(f)
                f_norm =  f_abs/(f_abs.max()/255.0) 
                fshift = np.fft.fftshift(f_norm) # componente de frequência no centro
                spectrum = np.log(1+np.abs(fshift)) # espectro de magnitude
                parameter=vgg16_feature_extraction(spectrum).reshape(-1,1)
                filenames.append(image_path)
                features.append(parameter)
                labels.append(label)
    return features,filenames,labels          


def load_images_from_folder_and_removebg(folder):
    labels_ = os.listdir(folder)
    features = []
    labels   = []
    filenames = []
    for i, label in enumerate(labels_):
        cur_path = folder + "/" + label 
        for image_path in glob.glob(cur_path + "/*"):
            img = cv2.imread(os.path.join(folder,image_path))
            if img is not None:
                output = remove(img)
                new_path = os.path.join(folder,image_path).replace('.jpg','bg-removal.png').replace('dataset_separado','dataset_rotulado_sem_bg')
                print(new_path)
                cv2.imwrite(new_path,output)   



#df,filenames = load_images_from_folder('C:\\Users\\lueli\\ProjetoPecem\\imagens_nao_rotuladas\\all')

#df,filenames = load_images_from_folder('C:\\Users\\lueli\\ProjetoPecem\\imagens_nao_rotuladas\\dataset_rotulado')
dataset_path = 'C:\\Users\\lueli\\ProjetoPecem\\imagens_nao_rotuladas\\dataset_rotulado\\dataset_separado'

#load_images_from_folder_and_removebg(dataset_path)

df,filenames,labels = load_images_from_folder_and_fourrier_and_transfer('C:\\Users\\lueli\\ProjetoPecem\\imagens_nao_rotuladas\\dataset_rotulado\\dataset_rotulado_sem_bg')

print(labels)

arr = np.array(df)
print(arr.shape)

arr2 = arr.reshape(40,4096)


kmeans = KMeans(n_clusters=4, random_state = 42).fit(arr2)
result = kmeans.labels_

strings = [str(x) for x in result]
print(strings)

targetNames = ['Excelente','Bom', 'Ruim', 'Pessimo']
targetNames = np.array(sorted(targetNames))


cm = confusion_matrix(labels, strings)
title='Confusion matrix'
cmap=plt.cm.Blues
normalize = True
import itertools
if normalize:
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print("Normalized confusion matrix")
else:
    print('Confusion matrix, without normalization')
tick_marks = [0,1,2,3]
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
plt.show()

