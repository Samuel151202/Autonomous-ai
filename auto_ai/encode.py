import numpy as np
from tensorflow.keras.utils import to_categorical
from sklearn import preprocessing

def convert_array(X_data):
    X_array = np.array(X_data)
    return X_array

def target_process(y_list):
    '''fonction qui process la target en appliquant en Labelencoder ainsi qu'un to_categorical
    et retourne la target processed'''
    label_encoder = preprocessing.LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_list)
    y_cat = to_categorical(y_encoded)
    return y_cat
