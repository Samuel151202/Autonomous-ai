from tensorflow import keras
from keras.utils import to_categorical
from PIL import Image, ImageDraw, ImageColor, ImageFont
import numpy as np
import os
import pandas as pd


def list_images_builder_png(directory='autonomous-ai/data/processed_images/'): #Takes a directory and return the list of filename in the directory removing the suffix (.jpg)
    list_images=[]
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        # checking if it is a file
        if os.path.isfile(f):
            list_images.append(filename.removesuffix('.png')) #don't include the suffix
    return list_images

def X_train_builder(directory='autonomous-ai/data/processed_images/'):
    dict_image_array = {}
    for img in list_images_builder_png(directory):
        image_train = Image.open(os.path.join(f'{directory}{img}.png'))
        image_train_normalized = image_train / 255.
        dict_image_array[img] = np.array(image_train_normalized)
    return dict_image_array

if __name__ == '__main__':
    print(X_train_builder())
