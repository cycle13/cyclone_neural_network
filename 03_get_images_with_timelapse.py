import os
from shutil import copyfile
import pandas
import numpy as np
import time
from datetime import datetime
import re
import matplotlib.pyplot as plt
import pickle


ORIGINAL_DIR = './images1/'
FINAL_DIR = './images/'
FILE_EXTS = ['.jpg', '.location.tsv', '.bboxes.tsv', '.bboxes.labels.tsv']
TIME_FMT = '%Y%m%d_%H%M'


# Function to copy files to the other directory
def copy_files(file_names):
    i = 1
    for item in file_names :
        for file_ext in FILE_EXTS:
            copyfile(ORIGINAL_DIR + item + file_ext, FINAL_DIR + item + file_ext)
        print(i)
        i+=1



# Funtion to find number of seconds since epoch
def epoch_time(time_string):
    result = datetime.strptime(time_string, TIME_FMT) - datetime(2000,1,1)
    return int(result.total_seconds())



# Get the names of the files that are annotated
file_names = os.listdir(ORIGINAL_DIR)
annotated = []
for item in file_names:
    if item.endswith('.location.tsv'):
        annotated.append(item.split('.')[0])
print(len(annotated))
# copy_files(annotated)
print(len(os.listdir(FINAL_DIR)))

#
#
# # Read in all download links
# with open('tutorial/cyclone.html', 'r') as f:
#     links = f.readlines()
#
# # Split download links into a dictionary with year as key
# links_by_year = {}
# for link in links:
#     year = '20' + link.split('/')[4][2:]
#     if year not in links_by_year.keys():
#         links_by_year[year] = [link]
#     else:
#         links_by_year[year].append(link)
#
#
#
# # Find out what are the time diffs available
# time_diffs = []
# for name in annotated:
#     print(name)
#     name_comps = name.split('_')
#     year = '20' + name_comps[0]
#     basin = name_comps[1]
#     cyc_name = name_comps[2]
#     timestamp = name_comps[3] + '_' + name_comps[4]
#
#
#     for item in links_by_year[year]:
#         dl_comps = item.split('/')
#         dl_name  = dl_comps[6].split('.')[1]
#         dl_timestamp = dl_comps[10].split('.')[0] + '_' + \
#                         dl_comps[10].split('.')[1]
#         # print(dl_timestamp_date)
#         # print(item)
#
#         if bool(re.match("^[0-9_]*$", dl_timestamp)):
#             if cyc_name == dl_name:
#                 tdelta = epoch_time(dl_timestamp) - epoch_time(timestamp)
#
#                 # Only add timestamps that are less than 6 hour differenc
#                 # if np.abs(tdelta) < 6*3600:
#                 time_diffs.append(tdelta)
#                 # print(item)

# pickle.dump(time_diffs, open('time_diff.p', 'wb'))
time_diffs =  pickle.load(open('time_diff.p', 'rb'))

time_diffs = np.array(time_diffs)/3600
binwidth = 6
plt.hist(time_diffs, bins=range(-24, 24, binwidth))
plt.title("Gaussian Histogram")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.show()
#
# fig = plt.gcf()
# plot_url = py.plot_mpl(fig, filename='mpl-basic-histogram')

#
#
# # Extract good download links
# good_dl = []
# total_of_good_cyclones = 0
#
# for year in storm_names.index:
#     # Cyclone name in IBTrACS
#     ib_names = storm_names[year]
#     # Cyclone name in download links
#     dl_names = [c.split('/')[6].split('.')[1] for c in links_by_year[year]]
#
#     # Extract cyclone with records in IBTrACS
#     cyclone_with_track = []
#     for cyclone in ib_names:
#         if cyclone in dl_names:
#             cyclone_with_track.append(cyclone)
#
#     # Get all storm data for the year in IBTrACS
#     storm_data = data[data['Season'] == year]
#
#     # Timestamps from download links
#     lc = [l.split('/') for l in links_by_year[year]]
#     available_timestamps = [l[10].split('.')[0] + l[10].split('.')[1] for l in lc]
#
#     #Extract good download links for each cyclone
#     for cyclone in cyclone_with_track:
#         # Timestamp from IBTrACS
#         timestamps = storm_data[storm_data['Name']==cyclone]['ISO_time'].tolist()
#         c_timestamps = [time.strftime("%Y%m%d%H%M",datetime.datetime.strptime(s, "%Y-%m-%d %H:%M:%S").timetuple()) for s in timestamps]
#
#         good_index = []
#         i = 0
#         lc_len = len(lc)
#         while i < lc_len:
#             if dl_names[i] == cyclone and available_timestamps[i] in c_timestamps:
#                 good_index.append(i)
#             i+=1
#
#         for i in good_index:
#             good_dl.append(links_by_year[year][i])
#     print(year, len(cyclone_with_track))
#     total_of_good_cyclones += len(cyclone_with_track)
#
#
# print(len(good_dl))
# print(total_of_good_cyclones)

# with open('download.txt','w') as f:
#     for item in good_dl:
#         f.writelines('%s' % item)
