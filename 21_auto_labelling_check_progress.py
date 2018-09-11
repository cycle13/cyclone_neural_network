import os
from shutil import copyfile
import pandas
import numpy as np
import time
import datetime
import re
import matplotlib.pyplot as plt
import pickle
from itertools import compress
import cv2
from helpers import *


IMG_DIR = './sequence_images/'
TIME_FMT = '%Y%m%d_%H%M'
LABEL_DIR = './images1/'

def copy_files(file_names):
    i = 1
    for item in file_names:
        print(i, end='\r', flush=True)
        copyfile(LABEL_DIR + item, IMG_DIR + item)
        i+=1

files = os.listdir(IMG_DIR)

images = []
labelled = []
centred = []
wind = []
for item in files:
    if item.endswith('.jpg'):
        images.append(item)
    elif item.endswith('.location.tsv'):
        centred.append(item.split('.')[0] + '.jpg')
    elif item.endswith('.bboxes.tsv'):
        labelled.append(item.split('.')[0] + '.jpg')
    elif item.endswith('.wind.tsv'):
        wind.append(item.split('.')[0] + '.jpg')

print('Number of images labelled: {}/{}'.format(len(labelled),len(images)))
print('Number of images with wind speed: {}/{}'.format(len(wind),len(images)))
print('Number of images with location: {}/{}'.format(len(centred),len(wind)))

f = [x in centred for x in wind]
f = list(compress(wind, f))
print(list(set(wind) - set(f)))

# # label_files = os.listdir(LABEL_DIR)
# annotated = []
# for item in label_files:
#     if item.endswith('labels.tsv'):
#         annotated.append(item.split('.')[0] + '.jpg')
# available_annotated = [x in annotated for x in images]
# available_annotated = list(compress(images, available_annotated))
# print('Number of labels available for the images: {}/{}'.format(len(available_annotated),len(images)))

# bbox = [(x.split('.')[0] + '.bboxes.tsv') for x in available_annotated]
# bbox_label = [(x.split('.')[0] + '.bboxes.labels.tsv') for x in available_annotated]
# print(bbox[0])
# print(bbox_label[0])

# copy_files(bbox)
# copy_files(bbox_label)


# Identify images with wind speed information
# data = pandas.read_csv('ibtracs.csv', low_memory=False, header=1, skiprows=[2])

# # Filter out only cyclones after 2001
# data = data[data['Season'] >= 2001]

# #01_ATL_ALLISON_20010606_0000.jpg
# i=0
# total_num_links = len(images)
# while i < total_num_links:
#     name_comps = images[i].split('.')[0].split('_')
#     timestamp = name_comps[3] + name_comps[4]
#     row = data[(data['Season']==int('20'+name_comps[0])) & (data['Name']==name_comps[2]) & 
#                     (data['ISO_time']==str(datetime.datetime.strptime(timestamp, "%Y%m%d%H%M")))]
#     wind_speed = row['Wind(WMO)'].values
#     if len(wind_speed) > 0:
#         with open(IMG_DIR + images[i].split('.')[0]+'.wind.tsv', 'w') as f:
#             f.write('{}\n'.format(wind_speed[0]))

#         lat = row['Latitude'].values[0]
#         lon = row['Longitude'].values[0]

#         with open(IMG_DIR + images[i].split('.')[0]+'.loc.tsv', 'w') as f:
#             f.write('{} {}\n'.format(lat, lon))
#     i+=1
#     print('{}/{}'.format(i,total_num_links),end='\r',flush=True)
