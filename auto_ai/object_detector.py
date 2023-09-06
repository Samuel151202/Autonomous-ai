import cv2
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt

### Preprocess image
def constrastLimit(image):
    img_hist_equalized = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    channels = cv2.split(img_hist_equalized)
    channels = list(channels) #convert to list to replace with the equalizeHist
    channels[0] = cv2.equalizeHist(channels[0]) #stretch out the intensity image of the image
    channels = np.array(channels) #convert back to np array
    img_hist_equalized = cv2.merge(channels)
    img_hist_equalized = cv2.cvtColor(img_hist_equalized, cv2.COLOR_YCrCb2BGR)
    return img_hist_equalized

def LaplacianOfGaussian(image):
    LoG_image = cv2.GaussianBlur(image, (3,3), 0)           # parameter
    gray = cv2.cvtColor( LoG_image, cv2.COLOR_BGR2GRAY)
    LoG_image = cv2.Laplacian( gray, cv2.CV_8U,3,3,2)       # parameter
    LoG_image = cv2.convertScaleAbs(LoG_image)
    return LoG_image

def binarization(image):
    thresh = cv2.threshold(image,32,255,cv2.THRESH_BINARY)[1]
    #thresh = cv2.adaptiveThreshold(image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
    return thresh

def preprocess_image(image):
    image = constrastLimit(image)
    image = LaplacianOfGaussian(image)
    image = binarization(image)
    return image

### Find signs
def removeSmallComponents(image, threshold=300):
    #find all your connected components (white blobs in your image)
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(image, connectivity=8)
    sizes = stats[1:, -1]; nb_components = nb_components - 1

    img2 = np.zeros((output.shape),dtype = np.uint8)
    #for every component in the image, you keep it only if it's above threshold
    for i in range(0, nb_components):
        if sizes[i] >= threshold:
            img2[output == i + 1] = 255
    return img2

def findContour(image):
    #find contours in the thresholded image
    cnts = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return cnts

def contourIsSign(perimeter, centroid, threshold): # Compute signature of contour

    result=[]
    for p in perimeter:
        p = p[0]
        distance = sqrt((p[0] - centroid[0])**2 + (p[1] - centroid[1])**2)
        result.append(distance) #distance between the centroid and the borders
    max_value = max(result) #retain max distance
    signature = [float(dist) / max_value for dist in result ] #ratio of each distance vs the max distance
    # Check signature of contour.
    temp = sum((1 - s) for s in signature) #each inverted distance summed up
    temp = temp / len(signature) #mean of the inverted distance
    if temp < threshold: # is  the sign
        return True, max_value + 2
    else:                 # is not the sign
        return False, max_value + 2

### Crop signs
def cropContour(image, center, max_distance):
    width = image.shape[1]
    height = image.shape[0]
    top = max([int(center[0] - max_distance), 0])
    bottom = min([int(center[0] + max_distance + 1), height-1])
    left = max([int(center[1] - max_distance), 0])
    right = min([int(center[1] + max_distance+1), width-1])
    #print(left, right, top, bottom)
    return image[left:right, top:bottom]

def findSigns(image, contours, threshold=0.6, distance_theshold=0.15):
    signs = []
    coordinates = []
    for c in contours[0]:
        # compute the center of the contour
        M = cv2.moments(c)
        if M["m00"] == 0:
            continue
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        is_sign, max_distance = contourIsSign(c, [cX, cY], 1-threshold)
        if is_sign and max_distance > distance_theshold:
            sign = cropContour(image, [cX, cY], max_distance)
            signs.append(sign)
            coordinate = np.reshape(c, [-1,2])
            top, left = np.amin(coordinate, axis=0)
            right, bottom = np.amax(coordinate, axis = 0)
            coordinates.append([(top-2,left-2),(right+1,bottom+1)])
    return signs, coordinates

### Final function to get list of signs, and a list of their coordinates

def panel_detector(path='data/test_images/French-Road-Signs.jpg'):
    image = cv2.imread(path) #that includes the conversion
    processed_image = preprocess_image(image)
    processed_image_ = removeSmallComponents(processed_image, threshold=300)
    contours = findContour(processed_image_)
    signs, coordinates = findSigns(image, contours, threshold=0.6, distance_theshold=0.15)
    return signs,coordinates,path

#Test the function and see image with signs highlighted

if __name__ == '__main__':
    filename='/home/parfait/code/Samuel151202/Autonomous-ai/data/images/u9lg1aqXm_UmrTn6eK4H_w.jpg'
    signs, coordinates,filename= panel_detector(filename)
    image = cv2.imread(filename)
    for coordinate in coordinates:
        image = cv2.rectangle(image,coordinate[0],coordinate[1],(0,255,0),3) #coordinate[0] #x1y1 #coordinate[1] #x2y2
        image = cv2.putText(image, 'steingate', (coordinate[0][0], coordinate[0][1] - 5),cv2.FONT_HERSHEY_SIMPLEX, 0.6, color=1, thickness=2)
    plt.imshow(image)
    plt.show()
