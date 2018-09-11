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
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import AveragePooling2D, Dense, Flatten, GlobalAveragePooling2D, BatchNormalization
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
from hyperas.distributions import uniform, choice
from hyperas import optim
from hyperopt import Trials, STATUS_OK, tpe
import copy
from keras.models import Model

# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"]="0"


class customValidationCallback(keras.callbacks.Callback):
    def on_train_begin(self, logs={}, min_delta=0, patience=5):
        self.losses = []
        self.val_error_means = []
        self.val_error_stds = []
        self.min_delta = min_delta
        self.patience = patience
        self.patience_left = patience

    def on_epoch_end(self, epoch, logs={}):
        self.losses.append(logs.get('loss'))
        prediction = self.model.predict(self.validation_data[0])
        val_error = np.abs(self.validation_data[1] - prediction)
        val_error = np.sqrt(np.dot(val_error**2, np.array([1,1])))
        current_error = np.mean(val_error)

        if len(self.val_error_means) > 0:
            delta = current_error - self.val_error_means[self.patience_left-self.patience-1]
            if delta > self.min_delta:
                self.patience_left -= 1
                if self.patience_left == 0:
                    self.model.stop_training = True

            else:
                # Reset patience_left if there is a decrease
                self.patience_left = self.patience
                self.model.save_weights('best_regressor.h5')
        self.val_error_means.append(current_error)
        self.val_error_stds.append(np.std(val_error))

BATCH_SIZE = 32
EPOCHS = 25
RESIZE_DIM = 256
RANDOM_CROP_DIM = 224


history = None

# Set up network training instance
base_model = DenseNet201(include_top=False, input_shape=(224,224,3),
                    weights='imagenet')

# x = AveragePooling2D((7, 7), name='avg_pool')(base_model.output)
x = GlobalAveragePooling2D()(base_model.output)
# x = Flatten()(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.3838)(x)
x = Dense(2, activation='linear')(x)

model = Model(inputs=base_model.input, outputs = x)

# model.load_weights('best_regressor.h5')

opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

model.compile(loss='mean_squared_error', optimizer=opt)




TRAIN_DATA = 'train_raw_data.p'
VAL_DATA = 'validate_raw_data.p'
RANDOM_CROP_NUMBER = 1
RESIZE_DIM = 256
RANDOM_CROP_DIM = 224


[x_train,y_train] = pickle.load(open(TRAIN_DATA, 'rb'))
[x_test,y_test] = pickle.load(open(VAL_DATA, 'rb'))
x_train = np.array(x_train)
x_test = np.array(x_test)
y_train = np.array([item[0] for item in y_train])
y_test = np.array([item[0] for item in y_test])

x_train, y_train, x_test, y_test = augment(x_train,
    y_train, x_test, y_test, ['all', 'colour'],
    RANDOM_CROP_NUMBER, RESIZE_DIM, RANDOM_CROP_DIM)

# Normalise input
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

y_train = y_train.astype('float32')
# Normalise output
y_test = y_test.astype('float32')
y_train /= 255
y_test /= 255

# Train the CNN
history = customValidationCallback()

model.fit(x_train, y_train,
    epochs=EPOCHS, batch_size=BATCH_SIZE,
    validation_data = (x_test, y_test),
    callbacks = [history]
)


SAVE_DIR = 'results/'

history_data = {
        'loss_history': history.losses,
        'val_error_means': history.val_error_means,
        'val_error_stds': history.val_error_stds,
    }
pickle.dump(history_data, open(SAVE_DIR + MODEL_NAME + '.p', 'wb'))


if __name__ == '__main__':
    import sys
    main(sys.argv[1:])
