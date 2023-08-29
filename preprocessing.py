import json
import os
import traceback
import logging
import numpy as np
from PIL import Image, ImageDraw, ImageColor, ImageFont
def list_images_builder(directory): #Takes a directory and return the list of filename in the directory removing the suffix (.jpg)
    list_images=[]
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        # checking if it is a file
        if os.path.isfile(f):
            list_images.append(filename.removesuffix('.jpg')) #don't include the suffix
    return list_images
def load_annotation(image_key): #return all annotations for each panel
    with open(os.path.join('autonomous-ai/data/annotations', '{:s}.json'.format(image_key)), 'r') as fid:
        anno = json.load(fid)
    return anno
def crop_image(list_images): #retrieve annotation and add annotations to the image
    count = 0
    for image in list_images:
        anno = load_annotation(image)
        with Image.open(os.path.join('autonomous-ai/data/images', '{:s}.jpg'.format(image))) as img:
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
                    processed_img.save(f"autonomous-ai/data/processed_images/{obj['key']}.png")
                    count+=1
    return count #return the image img and the bbox (coordinates of the road signs)
if __name__ == '__main__':
    #define a list of all images
    list_images = list_images_builder('autonomous-ai/data/images')
    # create a list of processed images
    crop_image(list_images)
