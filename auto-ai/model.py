
from tensorflow.keras import Sequential
#from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Dense, Conv2D, Flatten
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Conv2D,  Flatten

def initialize_model():
    ''' la fonction permet d'initialiser le model avec les différents layers ainsi que leurs paramètres'''
    model = Sequential([
        Conv2D(20,(4,4),activation='relu', input_shape=(244,244,1), name='inputlayer'),
        Conv2D(10,(3,3),activation='relu', name='layer2'),
        Conv2D(5,(3,3), activation='relu', name='layer3'),
        Flatten(),
        Dense(3, activation='relu', name='layer4'),
        Dense(3, activation='softmax', name='outputlayer')
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
                        epochs = 30,
                        batch_size = 20)
    return history

#def evaluate_model(model,X_test, y_test):
    '''la fonction retourne les resultats
    de prediction sur les données "test"'''
    #return model.evaluate(X_test, y_test)
