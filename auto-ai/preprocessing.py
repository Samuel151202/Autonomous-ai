import json
import os
import traceback
import logging
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageColor, ImageFont
from image_preprocessing import X_train_builder
PATH_ANNO = os.environ.get('PATH_ANNO')
PATH_IMAGE = os.environ.get('PATH_IMAGE')
PATH_PROC_IMAGE = os.environ.get('PATH_PROC_IMAGE')

def list_images_builder(directory): #Takes a directory and return the list of filename in the directory removing the suffix (.jpg)
    list_images=[]
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        # checking if it is a file
        if os.path.isfile(f):
            list_images.append(filename.removesuffix('.jpg')) #don't include the suffix
    return list_images

def load_annotation(image_key): #return all annotations for each panel
    with open(os.path.join(PATH_ANNO, '{:s}.json'.format(image_key)), 'r') as fid:
        anno = json.load(fid)
    return anno

def crop_image(list_images): #retrieve annotation and add annotations to the image
    count = 0
    for image in list_images:
        anno = load_annotation(image)
        with Image.open(os.path.join(PATH_IMAGE, '{:s}.jpg'.format(image))) as img:
        #get coordinates linked to each panel
            for obj in anno['objects']:
                x1 = obj['bbox']['xmin']
                y1 = obj['bbox']['ymin']
                x2 = obj['bbox']['xmax']
                y2 = obj['bbox']['ymax']
        #image processing + standardization
                try:
                    cropped_image = img.crop((x1, y1, x2, y2))
                    resizing_format = (244,244)
                    processed_img = cropped_image.resize(resizing_format)
                except Exception as e :
                    logging.error(traceback.format_exc()) # Logs the error appropriately
                    count+=1
                #save images in processed_images folder with counting
                if obj['label']!='other-sign':
                    processed_img.save(f"{PATH_PROC_IMAGE}/{obj['label']}_{obj['key']}.png")
                    count+=1
    return count #return the image img and the bbox (coordinates of the road signs)

def list_frequent_label(list_images, num_limit):
    complet_list = []
    if '.DS_Store' in list_images:
        list_images.remove('.DS_Store')
    for image in list_images:
        # Getting json file name from directory
        anno = load_annotation(image)
        # Do a list of pannel features in image
        for obj in anno['objects'] :
            if obj['label']!='other-sign':
                dict_ = {}
                dict_['image_name'] = image
                dict_compl = obj | dict_
                complet_list.append(dict_compl)
    # Do the DataFrame
    df=pd.DataFrame.from_dict(complet_list)
    return df['label'].value_counts()[df['label'].value_counts()>num_limit].index.tolist()

def getting_data(dict_image_array,list_frequent_labels): #dico is the dict returned by the func X_train_builder
    y_list=[]                                            #list_frequent_labels returned by the func list_frequent_label
    X_list=[]
    y=[]
    X=[]
    for key,value in dict_image_array.items():
        y.append(key.split('_')[0])
        X.append(value)
    for i in range(len(y)):
        if y[i] in list_frequent_labels:
            y_list.append(y[i])
            X_list.append(X)
    return [X_list,y_list] #Returns X_list np.array for model and y_list label for encoding before model


if __name__ == '__main__':
    #define a list of all images
    list_images = list_images_builder(PATH_IMAGE)
    print(len(list_images))
    # generate cropped images
    crop_image(list_images[0:100])

    #Get most frequent label to get data that makes sense for model
    list_frequent_labels=list_frequent_label(list_images[0:100],6)

    print(list_frequent_labels)

    # Getting X and y for models. y need to be encoded. X ready for model.
    X,y=getting_data(X_train_builder(),list_frequent_labels)

    print(X,y)
