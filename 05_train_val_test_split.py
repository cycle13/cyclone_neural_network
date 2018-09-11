import os
import datetime
import numpy as np
from collections import Counter
import cv2
from geopy.distance import lonlat, distance
import re
import pickle

IMG_DIR = 'sequence_images/'
TIME_FMT = '%Y%m%d_%H%M'
NEEDED_TIME = np.array([1,2,3,4]) * 6
VALIDATION_SET = ['06_CPAC_IOKE','16_EPAC_JIMENA','13_IO_PHAILIN',
                '16_WPAC_MALAKAS', '05_ATL_DELTA', '08_IO_NARGIS',
                '04_SHEM_JUBA']
TEST_SET = ['10_EPAC_CELIA', '06_EPAC_NORMAN', '05_ATL_WILMA',
            '06_EPAC_DANIEL', '15_IO_CHAPALA', '15_EPAC_OLAF',
            '01_SHEM_ALISTAIR']
TOP = 50
BOTTOM = 954
LEFT = 60
RIGHT = 964
RESIZE_DIM = 256

def coord_transform(data):
    r = re.compile("([0-9]+)([a-zA-Z]+)")
    m = r.match(data)

    numeric = int(m.group(1))
    direction = m.group(2)

    # Check if numeric is correct
    if numeric > 180:
        print("Invalid number range: " + data)
        return False

    if direction not in ['W','E','N','S']:
        print("Invalid direction: " + data)
        return False

    # Convert to pre-defined numbering system for coordinates
    if numeric == 0:
        return 0

    if direction in ['N','E']:
        return numeric

    if direction in ['W','S']:
        return -numeric



# Obtain names of all image files
files = os.listdir(IMG_DIR)
image_files = []
image_files_with_centre = []
for item in files:
    # Verify image has cyclone centre
    if item.endswith('.jpg'):
        image_files.append(item)

        if item.split('.')[0] + '.location.tsv' in files:
            image_files_with_centre.append(item)

print(len(image_files))
print(len(image_files_with_centre))

train_data = []
train_labels = []
train_names = []
val_data = []
val_labels = []
val_names = []
test_data = []
test_labels = []
test_names = []
i = 0

for item in image_files_with_centre:
    print(i,end='\r',flush=True)
    i+=1
    cyclone_name = item.split('.')[0]

    windspeed = 0
    with open(IMG_DIR + cyclone_name + '.wind.tsv', 'r') as f:
        windspeed = float(f.readlines()[0].strip())

    # Verify previous images exists
    name_comps = cyclone_name.split('_')
    timestring = '_'.join(name_comps[3:])
    timestamp = datetime.datetime.strptime(timestring, TIME_FMT)

    for delta in NEEDED_TIME:
        n_timestamp = timestamp - datetime.timedelta(hours=float(delta))
        expected_name = '_'.join(name_comps[:3]) + '_' + datetime.datetime.strftime(n_timestamp, TIME_FMT) + '.jpg'
        if expected_name not in image_files:
            continue

    if windspeed > 0:
        # Crop and resize image
        input_data = cv2.imread(IMG_DIR + cyclone_name + '.jpg')
        crop_img = input_data[TOP:BOTTOM, LEFT:RIGHT]
        resize_img = cv2.resize(crop_img,(RESIZE_DIM,RESIZE_DIM),
                                interpolation=cv2.INTER_LINEAR)

        # Identify labels for the image
        label = []

        with open(IMG_DIR + cyclone_name + '.location.tsv', 'r') as f:
            coordinates = list(map(int, f.readlines()[0].strip().split(' ')))
            latitude = round((coordinates[0]-TOP+1)/904*RESIZE_DIM - 1)
            longitude = round((coordinates[1]-LEFT+1)/904*RESIZE_DIM - 1)

            if latitude < 32 or latitude >= 224 or longitude < 32 or longitude >= 224:
                continue

            label.append([latitude, longitude])

        label.append(windspeed)

        # Length in pixels
        positions = open(IMG_DIR + cyclone_name + '.bboxes.tsv', 'r').readlines()[0].strip().split('\t')
        horizontal_length = (int(positions[2]) - int(positions[0]))/904*RESIZE_DIM
        vertical_length = (int(positions[3]) - int(positions[1]))/904*RESIZE_DIM

        # Coordinates of centre of cyclone
        coordinates = open(IMG_DIR + cyclone_name + '.loc.tsv', 'r').readlines()[0].strip().split(' ')
        cyclone_lat = float(coordinates[0])
        cyclone_lon = float(coordinates[1])

        # Find the scale of the image
        with open(IMG_DIR + cyclone_name + '.bboxes.labels.tsv', 'r') as f:
            coordinates = f.readlines()[0].strip()[1:-1].split(', ')
            top = coord_transform(coordinates[0][1:-1])
            bottom = coord_transform(coordinates[1][1:-1])
            left = coord_transform(coordinates[2][1:-1])
            right = coord_transform(coordinates[3][1:-1])
            horizontal_scale = distance(lonlat(*(left, cyclone_lat)), lonlat(*(right,cyclone_lat))).km / horizontal_length
            vertical_scale = distance(lonlat(*(cyclone_lon, top)), lonlat(*(cyclone_lon, bottom))).km /vertical_length
            label.append([horizontal_scale, vertical_scale])

        # Putting the data into the appropriate set
        if '_'.join(cyclone_name.split('_')[:3]) in VALIDATION_SET:
            val_data.append(resize_img)
            val_labels.append(label)
            val_names.append(cyclone_name)
        elif '_'.join(cyclone_name.split('_')[:3]) in TEST_SET:
            test_data.append(resize_img)
            test_labels.append(label)
            test_names.append(cyclone_name)
        else:
            train_data.append(resize_img)
            train_labels.append(label)
            train_names.append(cyclone_name)

print(len(train_labels))
print(len(val_labels))
print(len(test_labels))

with open('train_raw_data.p', 'wb') as f:
    pickle.dump([train_data,train_labels], f, protocol=pickle.HIGHEST_PROTOCOL)
with open('validate_raw_data.p', 'wb') as f:
    pickle.dump([val_data,val_labels], f, protocol=pickle.HIGHEST_PROTOCOL)
with open('test_raw_data.p', 'wb') as f:
    pickle.dump([test_data,test_labels], f, protocol=pickle.HIGHEST_PROTOCOL)
