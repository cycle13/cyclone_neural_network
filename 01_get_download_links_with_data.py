import pandas
import numpy as np
import time
import datetime

data = pandas.read_csv('ibtracs.csv', low_memory=False, header=1, skiprows=[2])

# Filter out only cyclones after 2001
data = data[data['Season'] >= 2001]

# Get unique cyclone name from ibtracs
storm_names = data.groupby('Season')['Name'].apply(np.unique)

# Read in all download links
with open('tutorial/cyclone.html', 'r') as f:
    links = f.readlines()

# Split download links into a dictionary with year as key
links_by_year = {}
for link in links:
    year = int('20' + link.split('/')[4][2:])
    if year not in links_by_year.keys():
        links_by_year[year] = [link]
    else:
        links_by_year[year].append(link)

# Extract good download links
good_dl = []
total_of_good_cyclones = 0

for year in storm_names.index:
    # Cyclone name in IBTrACS
    ib_names = storm_names[year]
    # Cyclone name in download links
    dl_names = [c.split('/')[6].split('.')[1] for c in links_by_year[year]]

    # Extract cyclone with records in IBTrACS
    cyclone_with_track = []
    for cyclone in ib_names:
        if cyclone in dl_names:
            cyclone_with_track.append(cyclone)
    
    # Get all storm data for the year in IBTrACS
    storm_data = data[data['Season'] == year]

    # Timestamps from download links
    lc = [l.split('/') for l in links_by_year[year]]
    available_timestamps = [l[10].split('.')[0] + l[10].split('.')[1] for l in lc]

    #Extract good download links for each cyclone 
    for cyclone in cyclone_with_track:
        # Timestamp from IBTrACS
        timestamps = storm_data[storm_data['Name']==cyclone]['ISO_time'].tolist()
        c_timestamps = [time.strftime("%Y%m%d%H%M",datetime.datetime.strptime(s, "%Y-%m-%d %H:%M:%S").timetuple()) for s in timestamps]

        good_index = []
        i = 0
        lc_len = len(lc)
        while i < lc_len:
            if dl_names[i] == cyclone and available_timestamps[i] in c_timestamps:
                good_index.append(i)
            i+=1

        for i in good_index:
            good_dl.append(links_by_year[year][i])
    print(year, len(cyclone_with_track))
    total_of_good_cyclones += len(cyclone_with_track)


print(len(good_dl))
print(total_of_good_cyclones)

with open('download.txt','w') as f:
    for item in good_dl:
        f.writelines('%s' % item)