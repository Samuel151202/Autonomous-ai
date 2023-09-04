### IMPORT CLASSIC ###
import numpy as np
import pandas as pd
import os
import json
from matplotlib import pyplot as plt

### IMPORT PROJECT ###
import cv2

import tensorflow as tf
from tensorflow import keras


### CONFIG CLASS ###
class Config:
    SEED = 42
    RATIO = 0.25

    DIM = (256, 256)

    VAL_SPLIT = 0.25
    BATCH_SIZE = 2
    EPOCHS = 10

    TARGET_COLS = []

    AUTOTUNE = tf.data.AUTOTUNE
    image_path = "raw_data/images/"
    annotation_path = "raw_data/annotations/"

    test_path = "raw_data/images/*.jpg"
    test_img = "raw_data/images/00-BnNquK24qWw8IAFyXJQ.jpg"
    test_anno = "raw_data/annotations/00-BnNquK24qWw8IAFyXJQ.json"


# Init Configuration
CFG = Config()


### METHODS ###


# Load Images, Resize & GreyScale
def load_img(path):
    img = cv2.imread(path)

    width = int(img.shape[1] * CFG.RATIO)
    height = int(img.shape[0] * CFG.RATIO)
    dim = (width, height)

    img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    img = tf.image.rgb_to_grayscale(img)
    return tf.cast(img, tf.float32)


def load_img_256(path):
    img = cv2.imread(path)
    dimensions = img.shape
    dim = CFG.DIM

    img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    img = tf.image.rgb_to_grayscale(img)
    img = tf.cast(img, tf.float32)
    return img, dimensions


# Load Annotations
# Return labels & bbox associated
def load_annotation(path):
    labels = []
    bboxs = []
    feature_keys = ["xmin", "ymin", "xmax", "ymax"]

    with open(path) as f:
        file = json.load(f)
    datas = file.get("objects")

    if datas:
        for data in datas:
            if data["label"] != "other-sign":
                labels.append(data["label"])
                bboxs.append(data["bbox"])
            else:
                # print('Wrong Label')
                pass

    labels = tf.strings.as_string(labels)
    bboxs = [[int(data[key]) for key in feature_keys] for data in bboxs]
    bboxs = tf.constant(bboxs, dtype=tf.int64)

    return labels, bboxs


# Load data, using methods above
# Return images, labels & bbox
def load_data(path):
    path = tf.convert_to_tensor(path)
    path = bytes.decode(path.numpy())
    filename = path.split("/")[-1].split(".")[0]

    image_path = os.path.join("raw_data", "images", f"{filename}.jpg")
    annotation_path = os.path.join("raw_data", "annotations", f"{filename}.json")

    images, dims = load_img_256(image_path)
    labels, bboxs = load_annotation(annotation_path)

    ratio_X = CFG.DIM[0] / dims[1]
    ratio_Y = CFG.DIM[1] / dims[0]
    xmin = int(bboxs[0][0] * ratio_X)
    ymin = int(bboxs[0][1] * ratio_Y)
    xmax = int(bboxs[0][2] * ratio_X)
    ymax = int(bboxs[0][3] * ratio_Y)

    bboxs = (xmin, ymin, xmax, ymax)
    bboxs = tf.constant(bboxs, dtype=tf.int64)

    return images, labels, bboxs


# Method to map all the data
def mappable_func(path):
    result = tf.py_function(load_data, [path], (tf.float32, tf.string, tf.int64))
    return result


# Buiding and Split the data
# Depends the size of data
def build_dataset(path):
    nb_data = len(list(tf.data.Dataset.list_files(path)))  # Get the size of the dataset

    data = (
        tf.data.Dataset.list_files(path)
        .shuffle(nb_data, reshuffle_each_iteration=False)
        .map(mappable_func)
        .batch(CFG.BATCH_SIZE, drop_remainder= True)
        .prefetch(CFG.AUTOTUNE)
    )

    return data


def split_data(data):
    nb_data = int(len(data) * CFG.VAL_SPLIT)
    train = data.take(nb_data)
    test = data.skip(nb_data)

    return train, test


if __name__ == "__main__":
    img, label, bbox = load_data(CFG.test_img)
    plt.imshow(img)
