"""
    This programme implements simple LeNet CNN on training to regress
    hurricane centre position.

    Script should be run as followed:

    'python 11_lenet.py input output option'

    + input: name of input data file
    + output: name of output log file
    + option: special option to perform on the data
        * normalisation
        * augmentation
"""

from __future__ import print_function
import keras
from keras.applications.densenet import DenseNet201
from keras.layers import AveragePooling2D, Dense, Flatten, GlobalAveragePooling2D, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from keras.utils import plot_model
import os
import pickle
import cv2
import numpy as np
from numpy import reshape
import random
from augmentation import augment
from keras.models import Model
from keras.utils import np_utils

# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"]="0"

class customValidationCallback(keras.callbacks.Callback):
        def on_train_begin(self, logs={}, min_delta=0, patience=5):
            self.losses = []
            self.min_delta = min_delta
            self.patience = patience
            self.patience_left = patience
            self.best_model = None
            self.val_f1s = []
            self.val_recalls = []
            self.val_precisions = []
            self.val_accuracy = []

        def on_epoch_end(self, epoch, logs={}):
            self.losses.append(logs.get('loss'))
            val_predict = self.model.predict(self.validation_data[0])
            val_predict = np.argmax(val_predict, axis=1)
            val_targ = np.argmax(self.validation_data[1],axis=1)
            _val_f1 = f1_score(val_targ, val_predict, average=None)
            _val_recall = recall_score(val_targ, val_predict, average=None)
            _val_precision = precision_score(val_targ, val_predict, average=None)
            _val_acc = accuracy_score(val_targ, val_predict)
            self.val_f1s.append(_val_f1)
            self.val_recalls.append(_val_recall)
            self.val_precisions.append(_val_precision)
            print(' - val_accuracy {}'.format( _val_acc))
            
            if len(self.val_accuracy) > 0:
                delta = _val_acc - self.val_accuracy[self.patience_left-self.patience-1]
                if delta < self.min_delta:
                    self.patience_left -= 1
                    if self.patience_left == 0:
                        self.model.stop_training = True

                else:
                    # Reset patience_left if there is a decrease
                    self.patience_left = self.patience
                    self.best_model = self.model.get_weights()
                    
            self.val_accuracy.append(_val_acc)



def cyclone_classification(wind_speed):
    # Wind speed is in knots
    if wind_speed >= 137:
        return 0
    if wind_speed <= 136 and wind_speed >= 113:
        return 1
    if wind_speed <= 112 and wind_speed >= 96:
        return 2
    if wind_speed <= 95 and wind_speed >= 83:
        return 3
    if wind_speed <= 82 and wind_speed >= 64:
        return 4
    if wind_speed <= 63 and wind_speed >= 34:
        return 5
    if wind_speed <= 33 and wind_speed > 0:
        return 6

    return None

# Main function
def main(args):

    # Hyper-parameters
    BATCH_SIZE = 32
    EPOCHS = 25
    # SAVE_DIR = '/vol/bitbucket/qn14/'
    # TRAIN_DATA = '/vol/bitbucket/qn14/train_raw_data.p'
    # VAL_DATA = '/vol/bitbucket/qn14/validate_raw_data.p'
    SAVE_DIR = 'results/'
    TRAIN_DATA = 'train_raw_data.p'
    VAL_DATA = 'validate_raw_data.p'
    MODEL_NAME = args[0]
    RANDOM_CROP_NUMBER = 1
    RESIZE_DIM = 256
    RANDOM_CROP_DIM = 224

    # Set up network training instance
    base_model = DenseNet201(include_top=False, input_shape=(224,224,3),
                        weights='imagenet')

    # x = AveragePooling2D((7, 7), name='avg_pool')(base_model.output)
    x = GlobalAveragePooling2D()(base_model.output)
    # x = Dense(1024)(x)
    x = Dense(512)(x)
    x = Activation('relu')(x)
    x = Dropout(0.38)(x)
    x = Dense(7, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs = x)

    opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

    model.compile(loss='categorical_crossentropy', optimizer=opt)


    # Prepare data for training
    [X_train,y_train] = pickle.load(open(TRAIN_DATA, 'rb'))
    [X_val,y_val] = pickle.load(open(VAL_DATA, 'rb'))
    
    X_train = np.array(X_train)
    X_val = np.array(X_val)

    y_train = np.array([cyclone_classification(item[1]) for item in y_train])
    y_val = np.array([cyclone_classification(item[1]) for item in y_val])


    X_train, y_train, X_val, y_val = augment(X_train,
        y_train, X_val, y_val, ['classification','colour'],
        RANDOM_CROP_NUMBER, RESIZE_DIM, RANDOM_CROP_DIM)

    print(X_train.shape)
    print(y_train.shape)
    print(X_val.shape)
    print(y_val.shape)

    # Normalise input
    n = X_train.shape[0]
    d1 = X_train[:n//2,:].astype('float32')
    print(d1.shape)
    d2 = X_train[n//2:,:].astype('float32')
    print(d2.shape)
    X_train = np.vstack((d1, d2))
    X_val = X_val.astype('float32')
    X_train /= 255
    X_val /= 255
    y_train = np_utils.to_categorical(y_train)
    y_val = np_utils.to_categorical(y_val)

    # Train the CNN
    history = customValidationCallback()
    model.fit(X_train, y_train,
        epochs=EPOCHS, batch_size=BATCH_SIZE,
        validation_data = (X_val, y_val),
        callbacks = [history]
    )

    history_data = {
        'f1': history.val_f1s,
        'recall': history.val_recalls,
        'precision': history.val_precisions,
        'accuracy': history.val_accuracy,
        'best_model': history.best_model
    }

    pickle.dump(history_data, open(SAVE_DIR + MODEL_NAME + '.p', 'wb'))

    history = None

if __name__ == '__main__':
    import sys
    main(sys.argv[1:])
