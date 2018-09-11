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
# from hyperas.distributions import uniform, choice
# from hyperas import optim
# from hyperopt import Trials, STATUS_OK, tpe
import copy
from keras.models import Model


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

model.load_weights('best_regressor.h5')

opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

model.compile(loss='mean_squared_error', optimizer=opt)


TRAIN_DATA = 'train_raw_data.p'
# VAL_DATA = 'validate_raw_data.p'
TEST_DATA = 'test_raw_data.p'
RANDOM_CROP_NUMBER = 1
RESIZE_DIM = 256
RANDOM_CROP_DIM = 224


[x_train,y_train] = pickle.load(open(TRAIN_DATA, 'rb'))
[x_test,y_test_r] = pickle.load(open(TEST_DATA, 'rb'))

x_train = np.array(x_train)
y_train = np.array([item[0] for item in y_train])

x_test = np.array(x_test)
y_test = np.array([item[0] for item in y_test_r])
y_test_scale = np.array([item[2] for item in y_test_r])
y_test_wind = np.array([item[1] for item in y_test_r])

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

_, _, x_test, y_test = augment(x_train,
    y_train, x_test, y_test, ['translate', 'colour'],
    RANDOM_CROP_NUMBER, RESIZE_DIM, RANDOM_CROP_DIM)

# Normalise input and output
x_test = x_test.astype('float32')
y_test = y_test.astype('float32')

pickle.dump(x_test, open('cyclone_test_result_x.p', 'wb'))
pickle.dump(y_test, open('cyclone_test_result_y.p', 'wb'))

x_test /= 255
y_test /= 255

results = model.predict_on_batch(x_test)
print(results.shape)

pickle.dump(results * 255, open('cyclone_test_result_p.p', 'wb'))

val_error_comps = np.abs(y_test - results)
val_error = np.sqrt(np.dot(val_error_comps**2, np.array([1,1])))
pixel_error = np.mean(val_error) *255

print(pixel_error)

val_error_actual = {'h5':[], 'h4':[], 'h3':[], 'h2':[], 'h1':[], 'td':[], 'ts':[]}
def cyclone_classification(wind_speed):
    # Wind speed is in knots
    if wind_speed >= 137:
        return 'h5'
    if wind_speed <= 136 and wind_speed >= 113:
        return 'h4'
    if wind_speed <= 112 and wind_speed >= 96:
        return 'h3'
    if wind_speed <= 95 and wind_speed >= 83:
        return 'h2'
    if wind_speed <= 82 and wind_speed >= 64:
        return 'h1'
    if wind_speed <= 63 and wind_speed >= 34:
        return 'ts'
    if wind_speed <= 33 and wind_speed > 0:
        return 'td'

    return None

i = 0 
while i < len(val_error_comps):
    e = val_error_comps[i] * 255 * y_test_scale[i] 
    val_error_actual[cyclone_classification(y_test_wind[i])].append(np.sqrt(e[0]**2+e[1]**2))
    i+= 1

for item in val_error_actual.keys():
    print(item)
    print(np.mean(val_error_actual[item]))
