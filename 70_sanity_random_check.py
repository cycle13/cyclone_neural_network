import pickle
import numpy as np
from augmentation import augment


TRAIN_DATA = 'train_raw_data.p'
VAL_DATA = 'validate_raw_data.p'
RANDOM_CROP_NUMBER = 1
RESIZE_DIM = 256
RANDOM_CROP_DIM = 224

# Prepare data for training
[X_train,y_train] = pickle.load(open(TRAIN_DATA, 'rb'))
[X_val,y_val] = pickle.load(open(VAL_DATA, 'rb'))
X_train = np.array(X_train)
X_val = np.array(X_val)

y_train = np.array([item[0] for item in y_train])
y_val = np.array([item[0] for item in y_val])


X_train, y_train, X_val, y_val = augment(X_train,
    y_train, X_val, y_val, ['all', 'colour'],
    RANDOM_CROP_NUMBER, RESIZE_DIM, RANDOM_CROP_DIM)

result = np.abs(np.ones(y_val.shape)*255/2 - y_val)
result = np.sqrt(np.dot(result**2, np.array([1,1])))

print(np.mean(result))
print(np.std(result))
