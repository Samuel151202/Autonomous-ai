import cv2
import numpy as np
import tensorflow as tf
import requests
from io import BytesIO
from PIL import Image, ImageDraw, ImageColor, ImageFont
import matplotlib.pyplot as plt
import pandas as pd

def gaussian_blur_canny(directory = '/data/test_images/French-Road-Signs.png'):
    img = cv2.imread(directory) #that includes the conversion
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # Convert the fram to grayscale for edge detection
    blurred = cv2.GaussianBlur(src=gray, ksize=(3, 5), sigmaX=0.5) # Apply Gaussian blur to reduce noise and smoothen edges
    image_edged = cv2.Canny(blurred, 400, 750) # Perform Canny edge detection
    return img, image_edged

def houghlines_coordinates(image_edged):
    #Drawlines to cover area to box
    lines = cv2.HoughLinesP(
        image=image_edged,
        rho=1,
        theta=(np.pi/180),
        threshold=20)

    return lines

def box_coordinates(lines):
    df = []
    for i in range(len(lines)):
        df.append(lines[i][0])
    df = pd.DataFrame(df)
    df = df.rename(columns={0: "x1", 1: "y1", 2: "x2", 3: "y2"})

    #get top left coordinates
    x1_y1 = []
    x1_y1.append(df["x1"].min())
    x1_y1.append(df["y1"].min())

    #get bottom right coordinates
    x2_y2 = []
    x2_y2.append(df["x1"].max())
    x2_y2.append(df["y1"].max())

    return x1_y1, x2_y2

def box_placer(img, x1_y1, x2_y2):
    object_identifier = cv2.rectangle(img,x1_y1,x2_y2,(0,255,0),3) #place the box
    return object_identifier

def identified_object_cropper(img, x1_y1, x2_y2):
    cropped_img = img[x1_y1[1]:x2_y2[1], x1_y1[0]:x2_y2[0]] #img[y1:y2, x1:x2] crop the image
    plt.imshow(cropped_img)
    plt.show()

def object_detector(directory = 'Autonomous-ai/data/test_images/French-Road-Signs.jpg'):
    img, image_edged = gaussian_blur_canny(directory)
    lines = houghlines_coordinates(image_edged)
    x1_y1, x2_y2 = box_coordinates(lines)
    identified_object_cropper(img, x1_y1, x2_y2)

if __name__ == '__main__':
    object_detector()
