# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 02:10:28 2022

@author: lueli
"""

# example of using the vgg16 model as a feature extraction model
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.applications.vgg16 import decode_predictions
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Model
from pickle import dump
import cv2

def vgg16_feature_extraction(image):
    dim = (224,224)
    with tf.device('/CPU:0'):
        # resize image
        resized_image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
        print(resized_image.shape)
        # convert the image pixels to a numpy array
        image = img_to_array(resized_image)
        # reshape data for the model
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        # prepare the image for the VGG model
        image = preprocess_input(image)
        # load model
        model = VGG16()
        
        
        # remove the output layer
        model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
        # get extracted features
        features = model.predict(image)
    return features