import os
import datetime
import numpy as np
import cv2
import re
from geopy.distance import lonlat, distance


IMG_DIR = 'sequence_images/'
TIME_FMT = '%Y%m%d_%H%M'
NEEDED_TIME = np.array([4,3,2,1]) * 6
VALIDATION_SET = ['06_CPAC_IOKE','16_EPAC_JIMENA','13_IO_PHAILIN',
                '16_WPAC_MALAKAS', '05_ATL_DELTA', '08_IO_NARGIS',
                '04_SHEM_JUBA']
TEST_SET = ['10_EPAC_CELIA', '06_EPAC_NORMAN', '05_ATL_WILMA',
            '06_EPAC_DANIEL', '15_IO_CHAPALA', '15_EPAC_OLAF',
            '01_SHEM_ALISTAIR']
TOP = 50
BOTTOM = 954
LEFT = 0
RIGHT = 1024
RESIZE_DIM = 224



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


files_all = os.listdir(IMG_DIR)
images = []
images_with_centres = []

for item in files_all:
    file_name = item.split('.')[0]
    if item.endswith('.jpg'):
        images.append(file_name)

        if file_name + '.location.tsv' in files_all:
            images_with_centres.append(file_name)

print('Total number of images: ', len(images))
print('Total number of images with centres identified: ', len(images_with_centres))

images_with_sequence = []
i = 0
for item in images_with_centres:
    temp_sequence = []
    print(i,end='\r',flush=True)
    i+=1

    file_name = item.split('.')[0]

    windspeed = 0
    with open(IMG_DIR + file_name + '.wind.tsv', 'r') as f:
        windspeed = float(f.readlines()[0].strip())

    # Verify previous images exists
    name_comps = file_name.split('_')
    timestring = '_'.join(name_comps[3:])
    timestamp = datetime.datetime.strptime(timestring, TIME_FMT)

    for delta in NEEDED_TIME:
        n_timestamp = timestamp - datetime.timedelta(hours=float(delta))
        expected_name = '_'.join(name_comps[:3]) + '_' + datetime.datetime.strftime(n_timestamp, TIME_FMT)
        if expected_name not in images:
            continue
        temp_sequence.append(expected_name)
    temp_sequence.append(item)

    if len(temp_sequence) == 5 and windspeed > 0:
        images_with_sequence.append(temp_sequence)

print('Images with sequence: ', len(images_with_sequence))

betweens = []
train_data = []
train_labels = []
train_names = []
val_data = []
val_labels = []
val_names = []
test_data = []
test_labels = []
test_names = []

j = 0
write = 5
for image_sequence in images_with_sequence:
    print(j,end='\r',flush=True)
    j+=1



    # Identify the coordinates to crop for the images in question
    tops = []
    bottoms = []
    lefts = []
    rights = []
    lat = None
    lon = None

    i = 0
    while i < 5:
        # Position of bounding box
        positions = open(IMG_DIR + image_sequence[i] + '.bboxes.tsv', 'r').readlines()[0].strip().split('\t')
        left_p = int(positions[0])
        top_p = int(positions[1])
        right_p = int(positions[2])
        bottom_p = int(positions[3])

        # Coordinates of bounding box
        coordinates = open(IMG_DIR + image_sequence[i] + '.bboxes.labels.tsv', 'r').readlines()[0].strip()[1:-1].split(', ')
        top_c = coord_transform(coordinates[0][1:-1])
        bottom_c = coord_transform(coordinates[1][1:-1])
        left_c = coord_transform(coordinates[2][1:-1])
        right_c = coord_transform(coordinates[3][1:-1])

        lon_diff = right_c - left_c
        if lon_diff < 0:
            lon_diff = 360 - lon_diff

        lat_diff = top_c - bottom_c

        lefts.append((left_c + (LEFT - left_p) * lon_diff / (right_p - left_p)))
        rights.append((right_c + (RIGHT - right_p) * lon_diff / (right_p - left_p)))
        bottoms.append((bottom_c - (BOTTOM - bottom_p) * lat_diff / (bottom_p - top_p)))
        tops.append((top_c - (TOP - top_p) * lat_diff / (bottom_p- top_p)))

        if i == 4:
            # Coordinates of centre of cyclone
            coordinates = open(IMG_DIR + image_sequence[i] + '.loc.tsv', 'r').readlines()[0].strip().split(' ')
            lat = float(coordinates[0])
            lon = float(coordinates[1])
        i+=1

    left_same = np.max(lefts)
    right_same = np.min(rights)
    top_same = np.min(tops)
    bottom_same = np.max(bottoms)

    between = True

    if bottom_same >= top_same:
        between = False
    if left_same * right_same > 0:
        if left_same > right_same:
            between = False
    if left_same < -90 and right_same >= 90:
        between = False

    if (0 <= left_same < -90) and (-90 <= right_same < 0):
        between = False


    if left_same * right_same > 0 or (left_same < 0 and right_same >= 0 ):
        if not (left_same <= lon <= right_same):
            between = False
    else:
        if lon >= 0:
            if not (left_same <= lon <= right_same + 360):
                between = False
        else:
            if not (left_same <= lon + 360 <= right_same + 360):
                between = False

    if not (bottom_same <= lat <= top_same):
        between = False

    betweens.append(between)

    input_sequence = []
    input_sequence_label = []

    # When crop is ok, determine the resolution of the resultant images
    if between:
        i = 0
        while i < 5:
            image = image_sequence[i]
            # Position of bounding box
            positions = open(IMG_DIR + image + '.bboxes.tsv', 'r').readlines()[0].strip().split('\t')
            left_p = int(positions[0])
            top_p = int(positions[1])
            right_p = int(positions[2])
            bottom_p = int(positions[3])

            # Coordinates of bounding box
            coordinates = open(IMG_DIR + image + '.bboxes.labels.tsv', 'r').readlines()[0].strip()[1:-1].split(', ')
            top_c = coord_transform(coordinates[0][1:-1])
            bottom_c = coord_transform(coordinates[1][1:-1])
            left_c = coord_transform(coordinates[2][1:-1])
            right_c = coord_transform(coordinates[3][1:-1])

            # Determine the position of the crop
            lon_diff = right_c - left_c
            if lon_diff < 0:
                lon_diff = 360 + lon_diff
            lat_diff = top_c - bottom_c


            left_diff = left_same - left_c
            if left_diff > 180:
                left_diff -= 360
            elif left_diff < -180:
                left_diff += 360
            left_t = left_p + (left_same - left_c) * lon_diff / (right_p - left_p)

            right_diff = right_same - right_c
            if right_diff > 180:
                right_diff -= 360
            elif right_diff < -180:
                right_diff += 360
            right_t = right_p + (right_same - right_c) * lon_diff / (right_p - left_p)

            top_t = top_p - (top_same - top_c) / lat_diff * (bottom_p - top_p)
            bottom_t = bottom_p - (bottom_same - bottom_c) / lat_diff * (bottom_p - top_p)

            print(image)
            print(top_same, bottom_same, left_same, right_same)
            print(top_c, bottom_c, left_c, right_c)
            print(top_p, bottom_p, left_p, right_p)
            print(top_t, bottom_t, left_t, right_t)
            # Crop and resize the image appropriately
            # print(image)

            img = cv2.imread(IMG_DIR + image + '.jpg')
            crop_img = img[int(top_t):int(bottom_t), int(left_t):int(right_t)]
            if write > 0:
                cv2.imwrite('illustrations/lstm_input'+str(i)+'.jpg',crop_img)
                write -= 1

            old_size = crop_img.shape[:2]
            ratio = float(RESIZE_DIM)/max(old_size)
            new_size = tuple([int(x*ratio) for x in old_size])
            resize_img = cv2.resize(crop_img, (new_size[1], new_size[0]), interpolation=cv2.INTER_LINEAR)
            delta_w = RESIZE_DIM - new_size[1]
            delta_h = RESIZE_DIM - new_size[0]
            top, bottom = delta_h//2, delta_h-(delta_h//2)
            left, right = delta_w//2, delta_w-(delta_w//2)
            color = [0, 0, 0]
            new_img = cv2.copyMakeBorder(resize_img, top, bottom, left, right, cv2.BORDER_CONSTANT,
                value=color)

            input_sequence.append(new_img)

            # Get labels from the last image of the sequence
            if i == 4:
                # Position of centre of cyclone
                coordinates = list(map(int, open(IMG_DIR + image + '.location.tsv', 'r').readlines()[0].strip().split(' ')))


                latitude = top + round((coordinates[0]-top_t+1)/old_size[0]*new_size[0] - 1)
                longitude = left + round((coordinates[1]-left_t+1)/old_size[1]*new_size[1] - 1)

                input_sequence_label.append([latitude, longitude])

                # Cyclone wind speed
                windspeed = 0
                with open(IMG_DIR + image + '.wind.tsv', 'r') as f:
                    windspeed = float(f.readlines()[0].strip())
                input_sequence_label.append(windspeed)

                # Coordinates of centre of cyclone
                coordinates = open(IMG_DIR + image + '.loc.tsv', 'r').readlines()[0].strip().split(' ')
                cyclone_lat = float(coordinates[0])
                cyclone_lon = float(coordinates[1])

                # Find the scale of the image
                horizontal_scale = distance(lonlat(*(left_same, cyclone_lat)), lonlat(*(right_same,cyclone_lat))).km / new_size[1]
                vertical_scale = distance(lonlat(*(cyclone_lon, top_same)), lonlat(*(cyclone_lon, bottom_same))).km / new_size[0]
                input_sequence_label.append([horizontal_scale, vertical_scale])
            i+=1

        # Putting the data into the appropriate set
        if '_'.join(image_sequence[4].split('_')[:3]) in VALIDATION_SET:
            val_data.append(input_sequence)
            val_labels.append(input_sequence_label)
            val_names.append(image_sequence[4])
        elif '_'.join(image_sequence[0].split('_')[:3]) in TEST_SET:
            test_data.append(input_sequence)
            test_labels.append(input_sequence_label)
            test_names.append(image_sequence[4])
        else:
            train_data.append(input_sequence)
            train_labels.append(input_sequence_label)
            train_names.append(image_sequence[4])


print('Number of sequences with centres ok: ', np.sum(betweens))
print(len(train_labels))
print(len(val_labels))
print(len(test_labels))

with open('train_lstm_data.p', 'wb') as f:
    pickle.dump([train_data,train_labels,train_names], f, protocol=pickle.HIGHEST_PROTOCOL)
with open('validate_lstm_data.p', 'wb') as f:
    pickle.dump([val_data,val_labels,val_names], f, protocol=pickle.HIGHEST_PROTOCOL)
with open('test_lstm_data.p', 'wb') as f:
    pickle.dump([test_data,test_labels, test_names], f, protocol=pickle.HIGHEST_PROTOCOL)
