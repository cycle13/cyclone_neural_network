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
from keras.applications.resnet50 import ResNet50
from keras.layers import AveragePooling2D, Dense, Flatten, GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import KFold
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

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0"

class customValidationCallback(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.val_error_means = []
        self.val_error_stds = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))

    def on_epoch_end(self, epoch, logs={}):
        prediction = self.model.predict(self.validation_data[0])
        val_error = np.abs(self.validation_data[1] - prediction)
        val_error = np.sqrt(np.dot(val_error**2, np.array([1,1])))
        self.val_error_means.append(np.mean(val_error))
        self.val_error_stds.append(np.std(val_error))  


      
# Main function
def main(args):

    # Hyper-parameters
    BATCH_SIZE = 128
    EPOCHS = 25
    SAVE_DIR = 'results'
    TRAIN_DATA = 'train_raw_data.p'
    VAL_DATA = 'validate_raw_data.p'
    MODEL_NAME = args[0]
    RANDOM_CROP_NUMBER = 2
    RESIZE_DIM = 256
    RANDOM_CROP_DIM = 224

    # Set up network training instance
    base_model = ResNet50(include_top=False, input_shape=(224,224,3), 
                        weights='imagenet')

    # x = AveragePooling2D((7, 7), name='avg_pool')(base_model.output)
    x = GlobalAveragePooling2D()(base_model.output)
    # x = Flatten()(x)
    x = Dense(1000, activation='relu')(x)
    x = Dense(2, activation='linear')(x)
    
    model = Model(inputs=base_model.input, outputs = x)

    opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

    model.compile(loss='mean_squared_error', optimizer=opt)


    # Prepare data for training
    [X_train,y_train] = pickle.load(open(TRAIN_DATA, 'rb'))
    [X_val,y_val] = pickle.load(open(VAL_DATA, 'rb'))
    X_train = np.array(X_train)
    X_val = np.array(X_val)
    
    y_train = np.array([item[0] for item in y_train])
    y_val = np.array([item[0] for item in y_val])
    
        
    X_train, y_train, X_val, y_val = augment(X_train, 
        y_train, X_val, y_val, args, 
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
    # X_train = X_train.astype('float32')
    X_val = X_val.astype('float32')
    X_train /= 255
    X_val /= 255

    # Normalise output
    y_train = y_train.astype('float32')
    y_val = y_val.astype('float32')
    y_train /= 255
    y_val /= 255

    # Train the CNN
    history = customValidationCallback()
    model.fit(X_train, y_train, 
        epochs=EPOCHS, batch_size=BATCH_SIZE,
        validation_data = (X_val, y_val),
        callbacks = [history]
    )

    history_data = {
        'loss_history': history.losses,
        'val_error_means': history.val_error_means,
        'val_error_stds': history.val_error_stds, 
    }

    model.save(MODEL_NAME + '.h5')


    pickle.dump(history_data, open(SAVE_DIR + '/' + MODEL_NAME + '.p', 'wb'))

    history = None

if __name__ == '__main__':
    import sys
    main(sys.argv[1:])
