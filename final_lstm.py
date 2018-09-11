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

import keras
from keras.applications.densenet import DenseNet201
from keras.layers import Activation, AveragePooling2D, Dense, Flatten, GlobalAveragePooling2D, LSTM, TimeDistributed, Dropout
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
from keras.models import Model, Sequential
from keras.utils import np_utils
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler


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

     
# Hyper-parameters
SAVE_DIR = 'results'
TRAIN_DATA = 'train_lstm_data.p'
VAL_DATA = 'validate_lstm_data.p'
TEST_DATA = 'test_lstm_data.p'

# Prepare data for training
[X_train,y_train, _ ] = pickle.load(open(TRAIN_DATA, 'rb'))
[X_val,y_val, _ ] = pickle.load(open(VAL_DATA, 'rb'))
[X_test,y_test, _ ] = pickle.load(open(TEST_DATA, 'rb'))
# X_val = np.array(X_val)
# X_test = np.array(X_test)

y_val = np.array([cyclone_classification(item[1]) for item in y_val])
y_test = np.array([cyclone_classification(item[1]) for item in y_test])
y_train = np.array([cyclone_classification(item[1]) for item in y_train])

# # Normalise input
# X_test = X_test.astype('float32')
# X_test /= 255

# base_model = DenseNet201(include_top=False, input_shape=(224,224,3),
#                     weights='imagenet')
# x = GlobalAveragePooling2D(name='intermediate_output')(base_model.output)
# x = Dense(512)(x)
# x = Activation('relu')(x)
# x = Dense(7, activation='softmax')(x)
# cnn = Model(inputs=base_model.input, outputs = x)
# cnn.load_weights('dense_best.h5')

# cnn_i = Model(inputs = cnn.input, outputs=cnn.get_layer('intermediate_output').output)

# X_test_new = []
# for item in X_test:
#     X_test_new.append(cnn_i.predict_on_batch(item))
# X_test = None
# X_test = np.array(X_test_new)

# # pickle.dump(X_val, open('lstm_val.p', 'wb'))
# pickle.dump(X_test, open('lstm_test.p', 'wb'))




# class customValidationCallback(keras.callbacks.Callback):
#     def on_train_begin(self, logs={}, min_delta=0, patience=10):
#         self.losses = []
#         self.min_delta = min_delta
#         self.patience = patience
#         self.patience_left = patience
#         self.best_model = None
#         self.val_f1s = []
#         self.val_recalls = []
#         self.val_precisions = []
#         self.val_accuracy = []

#     def on_epoch_end(self, epoch, logs={}):
#         self.losses.append(logs.get('loss'))
#         val_predict = self.model.predict(self.validation_data[0])
#         val_predict = np.argmax(val_predict, axis=1)
#         val_targ = np.argmax(self.validation_data[1],axis=1)
#         _val_f1 = f1_score(val_targ, val_predict, average=None)
#         _val_recall = recall_score(val_targ, val_predict, average=None)
#         _val_precision = precision_score(val_targ, val_predict, average=None)
#         _val_acc = accuracy_score(val_targ, val_predict)
#         self.val_f1s.append(_val_f1)
#         self.val_recalls.append(_val_recall)
#         self.val_precisions.append(_val_precision)

#         print(' - val_accuracy {}'.format( _val_acc))
    
#         if len(self.val_accuracy) > 0:
#             delta = _val_acc - self.val_accuracy[self.patience_left-self.patience-1]
#             if delta < self.min_delta:
#                 self.patience_left -= 1
#                 if self.patience_left == 0:
#                     self.model.stop_training = True

#             else:
#                 # Reset patience_left if there is a decrease
#                 self.patience_left = self.patience
#                 # self.best_model = self.model.get_weights()
#                 self.model.save_weights('lstm_best_final1.h5')

#         self.val_accuracy.append(_val_acc)

X_val = pickle.load(open('lstm_val.p', 'rb'))
X_test = pickle.load(open('lstm_test.p', 'rb'))
X_train = pickle.load(open('lstm_train.p', 'rb'))

y_val = np_utils.to_categorical(y_val)
y_test = np_utils.to_categorical(y_test)
y_train = np_utils.to_categorical(y_train)


# Set up network training instance
# Define LSTM model
model = Sequential()
model.add(LSTM(2048, return_sequences=True,  input_shape=(5,1920),
                dropout=0.5))
model.add(LSTM(2048, return_sequences=True, dropout=0.5))
model.add(LSTM(2048, return_sequences=False, dropout=0.5))
model.add(Dense(512, activation='relu'))
model.add(Dense(7, activation='softmax'))

# opt = keras.optimizers.adam(decay=1e-6, amsgrad=True)

# model.compile(loss='categorical_crossentropy', optimizer=opt)

# EPOCHS = 50
# BATCH_SIZE = 128

# history = customValidationCallback()

# model.fit(X_train, y_train,
#         epochs=EPOCHS, batch_size=BATCH_SIZE,
#         validation_data = (X_val, y_val),
#         callbacks = [history]
# )


model.load_weights('lstm_best_final.h5')

results = model.predict_on_batch(X_test)


# history_data = {
#     'f1': history.val_f1s,
#     'recall': history.val_recalls,
#     'precision': history.val_precisions,
#     'accuracy': history.val_accuracy,
#     'best_model': history.best_model
# }

# pickle.dump(history_data, open('results_lstm_best_final1.p', 'wb'))




predict = np.argmax(results, axis=1)
target = np.argmax(y_test, axis=1)
f1 = f1_score(target, predict, average=None)
recall = recall_score(target, predict, average=None)
precision = precision_score(target, predict, average=None)
acc = accuracy_score(target, predict)
con_mat = confusion_matrix(target, predict)
print(f1)
print(recall)
print(precision)
print(acc)
print(con_mat)


# # Train the LSTM network
# model.fit(X_train, y_train, 
#     epochs=EPOCHS, batch_size=BATCH_SIZE,
#     validation_data = (X_val, y_val),
#     callbacks = [history]
# )


# model.save(MODEL_NAME + '.h5')


# pickle.dump(history_data, open(SAVE_DIR + '/' + MODEL_NAME + '.p', 'wb'))

# history = None


