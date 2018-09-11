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

ORIGINAL_DIR = './images1/'
FINAL_DIR = './sequence_images/'
FILE_EXTS = ['.jpg', '.location.tsv', '.bboxes.tsv', '.bboxes.labels.tsv']
TIME_FMT = '%Y%m%d_%H%M'

# Function to copy files to the other directory
def copy_files(file_names):
    i = 1
    for item in file_names:
        print(i, end='\r', flush=True)
        copyfile(ORIGINAL_DIR + item, FINAL_DIR + item)
        i+=1


dl_links = []
with open('download_sequence.txt','r') as f:
    dl_links = f.readlines()

f_names = []
for item in dl_links:
    fc = item.split('/')
    year = fc[4][2:]
    basin = fc[5]
    cyclone_name = fc[6].split('.')[1]
    date = fc[10].split('.')[0]
    t = fc[10].split('.')[1]
    filename = year + '_' + basin + '_' + cyclone_name + '_' + date + '_' + t + '.jpg'
    f_names.append(filename)

available = os.listdir(ORIGINAL_DIR)
available_links = [x in available for x in f_names]
available_links = list(compress(f_names, available_links))
print('Number of links available on PC: {}/{}'.format(len(available_links),len(f_names)))

need_download = list(set(f_names) - set(available_links))
need_download_links = []
for item in need_download:
    need_download_links.append(dl_links[f_names.index(item)])

print(len(list(need_download_links)))
copy_files(available_links)


with open('need_download.txt','w') as f:
    for item in need_download_links:
        f.writelines('%s' % item)
