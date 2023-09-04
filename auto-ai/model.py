
from tensorflow.keras import Sequential
#from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Dense, Conv2D, Flatten
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Conv2D,  Flatten, MaxPooling2D, BatchNormalization , Dropout

def initialize_model(X,y_cat):
    ''' la fonction permet d'initialiser le model avec les différents layers ainsi que leurs paramètres'''
    model = Sequential([

        Conv2D(32, (3, 3), activation='relu', input_shape=X.shape[1:], name='conv1'),
        BatchNormalization(),
        MaxPooling2D((2, 2), name='maxpool1'),
        Conv2D(64, (3, 3), activation='relu', name='conv2'),
        BatchNormalization(),
        MaxPooling2D((2, 2), name='maxpool2'),
        Conv2D(128, (3, 3), activation='relu', name='conv3'),
        BatchNormalization(),
        MaxPooling2D((2, 2), name='maxpool3'),
        Flatten(),
        Dense(256, activation='relu', name='dense1'),
        Dropout(0.5),
        Dense(128, activation='relu', name='dense2'),
        Dropout(0.3),
        Dense(y_cat.shape[1], activation='softmax', name='outputlayer')
    ])

    return model

def compile_model(model):
    ''' la fonction correspond à la compilation du model qui
    correspond au choix des metrics que nous voulons pour la fonction '''
    model.compile( loss ='categorical_crossentropy',
                    optimizer='adam',
                    metrics = ['accuracy'])
    return model

def fit_model(model,X,y):
    ''' la fonction retourne
    les resultats du model fitter sur les données d'entrainement '''
    #es = EarlyStopping(patience = 5, verbose =2)
    history = model.fit(X, y,
                        validation_split = 0.3,
                        #callbacks = [es],
                        epochs = 200,
                        batch_size = 50)
    return history

def evaluate_model(model,X_test):
    '''la fonction retourne les resultats
    de prediction sur les données "test"'''
    return model.evaluate(X_test)
