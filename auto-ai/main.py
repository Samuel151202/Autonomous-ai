
from image_preprocessing import *
from preprocessing import *
from model import *
from encode import *

nbr_image=10000
nbr_limit = 200
#liste des images dans le dossier images
list_images = list_images_builder(PATH_IMAGE)

#retourne les cropped images
crop_image(list_images[0:nbr_image])

#renvoie le dictionnaire avec le y en target et le x en value
data_dict = X_train_builder(PATH_PROC_IMAGE)

#liste des label les plus fréquents dans la liste des images
# avec un nombre limit d'apparition
list_frequent_labels=list_frequent_label(list_images[0:nbr_image],nbr_limit)

''' fonction générale qui permet de récupérer le la X_list des images
ainsi que la y_list des différentes classes il prend en argument le data_dict
et la liste des label fréquent'''
X_list,y_list=getting_data(data_dict,list_frequent_labels) #dico is the dict returned by the func X_train_builder
#list_frequent_labels returned by the func list_frequent_label

#retourne le X en array
X = convert_array(X_list)
#retourne y processed
y_cat = target_process(y_list)

#retourne les resultat du model
model = initialize_model(X,y_cat)
model_compile =compile_model(model)
history = fit_model(model_compile,X,y_cat)
print(history.history)
