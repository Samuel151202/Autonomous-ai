import json
import os
import traceback
import logging
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageColor, ImageFont
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
                except Exception as e :
                    logging.error(traceback.format_exc()) # Logs the error appropriately
                    cropped_image=np.nan
                resizing_format = (244,244)
                processed_img = cropped_image.resize(resizing_format)
                #save images in processed_images folder with counting
                if obj['label']!='other-sign':
                    processed_img.save(f"{PATH_PROC_IMAGE}{obj['key']}.png")
                    count+=1
    return count #return the image img and the bbox (coordinates of the road signs)

def building_dataframe(list_images):
    complet_list = []
    if '.DS_Store' in list_images:
        list_images.remove('.DS_Store')

    for image in list_images:
        # Getting json file name from directory
        anno = load_annotation(image)
        # Do a list of pannel features in image
        for obj in anno['objects'] :
            if obj['label']!='other-sign':
                complet_list.append(obj)
    # Do the DataFrame
    df=pd.DataFrame.from_dict(complet_list)
    return df

if __name__ == '__main__':
    #define a list of all images
    list_images = list_images_builder(PATH_IMAGE)
    # create a list of processed images
    building_dataframe(list_images)
