from auto_ai.image_preprocessing import *
from auto_ai.preprocessing import *
from auto_ai.object_detector import *
from auto_ai.registry import *
import matplotlib.pylab as plt
from auto_ai.params import LISTE_LABEL
from PIL import Image, ImageDraw, ImageColor, ImageFont

def preproc(X_list):
    X_test=[]
    for i in range(len(X_list)):
        resizing_format = (64,64)
        processed_img = X_list[i].resize(resizing_format)
        image_bw_array = grayscale_n_array_converter(processed_img) #converts it to b&w using function below
        normalized_image_bw_array = image_bw_array / 255. #normalizes the image
        X_test.append(normalized_image_bw_array) #add to the list
    return X_test

def predict(X_test,coordinates,seuil):
    prob_liste=[]
    class_liste=[]
    coordinates_list=[]
    model=load_model()
    y_pred=model.predict(np.array(X_test))
    class_list=np.argmax(y_pred,1)
    prob=np.max(y_pred,1)
    for i in range(len(prob)):
        if prob[i]>seuil:
            prob_liste.append(prob[i])
            class_liste.append(class_list[i])
            coordinates_list.append(coordinates[i])
    return prob_liste,class_liste,coordinates_list

def full_preproc(X_list,coordinates,seuil=0.7,liste_label=LISTE_LABEL):
        label_list=[]
        sorted_list=sorted(liste_label)
        X_test=preproc(X_list)
        prob_list,class_list,coordinates=predict(X_test,coordinates,seuil)
        for i in class_list:
            label_list.append(sorted_list[i])
        return label_list,coordinates

def test_builder(filename):
    signs,coordinates,img_full=panel_detector(filename)
    Img=Image.open(img_full)
    X_list=[]
    for coordinate in coordinates:
        x1 = coordinate[0][0]
        y1 = coordinate[0][1]
        x2 = coordinate[1][0]
        y2 = coordinate[1][1]
        try:
            cropped_image = Img.crop((x1, y1, x2, y2))
            resizing_format = (64,64)
            processed_img = cropped_image.resize(resizing_format)
            X_list.append(processed_img)
        except Exception as e :
            logging.error(traceback.format_exc()) # Logs the error appropriately
            print('crop error')
    return X_list,coordinates

def demo(filename):
    X_list,coordinates= test_builder(filename)
    label_list,coordinates=full_preproc(X_list,coordinates,seuil=0.7,liste_label=LISTE_LABEL)
    image = cv2.imread(filename)
    for i in range(len(coordinates)):
        image = cv2.rectangle(image,coordinates[i][0],coordinates[i][1],(0,255,0),3) #coordinate[0] #x1y1 #coordinate[1] #x2y2
        image = cv2.putText(image,label_list[i], (coordinates[i][0][0], coordinates[i][0][1] - 5),cv2.FONT_HERSHEY_SIMPLEX, 0.6, color=0, thickness=2)
    plt.imshow(image)
    plt.show()



if __name__ == '__main__':
    filename='/home/parfait/code/Samuel151202/Autonomous-ai/data/test_images/Panneau-Stops.jpg'
    # X_list=test_builder(filename)
    # liste=full_preproc(X_list,0.7)
    # print(liste)

    demo(filename)



    #--------------.-------------------------------------------------------------#

 # file='/home/parfait/code/Samuel151202/Autonomous-ai/data/images/__2Myhq3esroCZGhOg_DdQ.jpg'

    # crop_img,coordinates=panel_detector(file)
    # print(crop_img[0])
    # liste=[]
    # for i in range(len(crop_img)):
    #     if crop_img[i]:
    #         img=cv2.resize(crop_img[0:5],(64,64),interpolation= cv2.INTER_LINEAR)
    #         liste.append(img)
    # plt.imshow(liste[0])
    # plt.show()

     # signs,coordinates=panel_detector(file)
    # file = cv2.imread(file)
    # for coordinate in coordinates:
    #     file = cv2.rectangle(file,coordinate[0],coordinate[1],(0,255,0),3)
    # plt.imshow(file)
    # plt.show()
