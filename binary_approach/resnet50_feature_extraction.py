import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
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
        # prepare the image for the ResNet50 model
        image = preprocess_input(image)
        # load model
        model = ResNet50(weights = 'imagenet', include_top = False)
        # get extracted features
        
        features = model.predict(image)
    return features