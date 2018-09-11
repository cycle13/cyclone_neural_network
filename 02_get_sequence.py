import os
from shutil import copyfile
import pandas
import numpy as np
import time
import datetime
import re
import matplotlib.pyplot as plt
import pickle


TIME_FMT = '%Y%m%d_%H%M'
NEEDED_TIME = np.array([1,2,3,4]) * 3600 * 6

# # Funtion to find number of seconds since epoch
def epoch_time(time_string):
    result = datetime.datetime.strptime(time_string, TIME_FMT) - datetime.datetime(2000,1,1)
    return int(result.total_seconds())


# Read in all download links
with open('tutorial/cyclone.html', 'r') as f:
    links = f.readlines()

# Split download links into a dictionary with year as key
links_by_year = {}
for link in links:
    link_comps = link.split('/')
    year = '20' + link_comps[4][2:]
    name = link_comps[6].split('.')[1]
    timestamp = link_comps[10].split('.')[0] + '_' + link_comps[10].split('.')[1]

    if link_comps[10].split('.')[1] is not '':
        if bool(re.match("^[0-9_]*$", timestamp)):
            # print(link)
            if year not in links_by_year.keys():
                links_by_year[year] = {name:[link]}

            else:
                if name not in links_by_year[year].keys():
                        links_by_year[year][name] = [link]
                else:
                    links_by_year[year][name].append(link)


cyclone_image_by_cat = pickle.load(open('dl_cat.p', 'rb'))

dl_links = []

for category in ['h5', 'h4', 'h3', 'h2', 'h1', 'ts', 'td']:
    data = cyclone_image_by_cat[category]
    print('{}: {} cyclones'.format(category, len(data)))

    # 'https://www.nrlmry.navy.mil/tcdat/tc17/WPAC/03W.MUIFA/ir/geo/1km_bw/20170429.0000.himawari8.x.ir1km_bw.03WMUIFA.20kts-1007mb-213N-1403E.100pc.jpg
    i = 0
    good_dl = 0
    total_num_links = len(data)
    while i < total_num_links and good_dl < 800:
        time_diffs = []
        s_links = []

        cyc_comps = data[i].split('/')
        cyc_year = '20' + cyc_comps[4][2:]
        cyc_name = cyc_comps[6].split('.')[1]
        cyc_timestamp = cyc_comps[10].split('.')
        cyc_timestamp = cyc_timestamp[0] + '_' + cyc_timestamp[1]

        for o_link in links_by_year[cyc_year][cyc_name]:
            o_cyc_comps = o_link.split('/')
            o_cyc_timestamp = o_cyc_comps[10].split('.')
            o_cyc_timestamp = o_cyc_timestamp[0] + '_' + o_cyc_timestamp[1]
            time_diffs.append(epoch_time(cyc_timestamp) - epoch_time(o_cyc_timestamp))
            s_links.append(o_link)

        if all(x in time_diffs for x in NEEDED_TIME):
            dl_links.append(data[i])
            for t in NEEDED_TIME:
                dl_links.append(s_links[time_diffs.index(t)])
            good_dl += 1
        i+=1
        print('{}/{}'.format(i,total_num_links),end='\r',flush=True)
        
    print('---------------------Done-------------------------')
    print(good_dl)

print(len(dl_links))
dl_links = list(set(dl_links))
print(len(dl_links))

with open('download_sequence.txt','w') as f:
    for item in dl_links:
        f.writelines('%s' % item)