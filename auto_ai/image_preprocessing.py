from tensorflow import keras
from keras.utils import to_categorical
from PIL import Image, ImageDraw, ImageColor, ImageFont, ImageOps
import numpy as np
import os
import pandas as pd
import cv2
PATH_ANNO = os.environ.get('PATH_ANNO')
PATH_IMAGE = os.environ.get('PATH_IMAGE')
PATH_PROC_IMAGE = os.environ.get('PATH_PROC_IMAGE')

def list_images_builder_png(directory=PATH_PROC_IMAGE): #Takes a directory and return the list of filename in the directory removing the suffix (.jpg)
    list_images=[]
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        # checking if it is a file
        if os.path.isfile(f):
            list_images.append(filename.removesuffix('.png')) #don't include the suffix
    return list_images

def grayscale_n_array_converter(image_train):
    image_file_array = np.array(image_train) #converting to an array (244, 244)
    grayscaled_image = cv2.cvtColor(image_file_array, cv2.COLOR_BGR2GRAY)
    grayscaled_image_reshaped = np.expand_dims(grayscaled_image, axis = -1) #forcing the shape (244, 244, 1)
    return grayscaled_image_reshaped

def X_train_builder(directory=PATH_PROC_IMAGE):
    dict_image_array = {}
    for img in list_images_builder_png(directory):
        image_train = Image.open(os.path.join(f'{directory}/{img}.png')) #finds the image
        image_bw_array = grayscale_n_array_converter(image_train) #converts it to b&w using function below
        normalized_image_bw_array = image_bw_array / 255. #normalizes the image
        dict_image_array[img] = np.array(normalized_image_bw_array) #add to the dictionary
    return dict_image_array # On a les y en keys et les y en values

if __name__ == '__main__':
    X_train_builder()
