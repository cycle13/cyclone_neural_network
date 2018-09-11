import pickle

TRAIN_DATA = '/vol/bitbucket/qn14/train_raw_data.p'

[X_train,y_train] = pickle.load(open(TRAIN_DATA, 'rb'))

scale = []
for item in y_train:
    scale.extend(item[2])
print(scale)
import numpy as np
print(np.mean(scale))