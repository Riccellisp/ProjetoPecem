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


# Biblioteca utilizada para feature importance
from sklearn.ensemble import RandomForestClassifier
# Biblioteca utilizada para normalização dos dados
from sklearn.preprocessing import StandardScaler
# Biblioteca para pca plot
from sklearn.decomposition import PCA
import matplotlib

x = StandardScaler().fit_transform(arr2)

pca = PCA(n_components=2)
principal_components = pca.fit_transform(x)
print ('Variância explicada acumulada',pca.explained_variance_ratio_.cumsum())
principal_df = pd.DataFrame(data = principal_components
              , columns = ['principal component 1', 'principal component 2'])

final_pca_df = pd.concat([principal_df, pd.DataFrame(labels)], axis = 1)
final_pca_df.columns = ['principal component 1', 'principal component 2', 'label']
fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)
targets = [0, 1]
colors = ['r', 'g','b','y']
for target, color in zip(targets,colors):
    indices_to_keep = final_pca_df['label'] == target
    ax.scatter(final_pca_df.loc[indices_to_keep, 'principal component 1']
                , final_pca_df.loc[indices_to_keep, 'principal component 2']
                , c = color
                , s = 50)
ax.legend(targets)
ax.grid() 
