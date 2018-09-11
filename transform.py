import os
import numpy as np
import re
import pandas
import time
import datetime



IMG_DIR = 'images1'



files = os.listdir(IMG_DIR)

labelled = [] 

for item in files:
    if item.endswith('.bboxes.labels.tsv'):
        labelled.append(item)

dataframe = pandas.read_csv('ibtracs.csv', low_memory=False, header=1, skiprows=[2])
dataframe['TimeStamp'] = [time.strftime("%Y%m%d%H%M",datetime.datetime.strptime(s, "%Y-%m-%d %H:%M:%S").timetuple()) for s in dataframe['ISO_time']]
  

for item in labelled:
    print(item)

    name_components = item.split('_')
    year = int('20' + name_components[0])
    data = dataframe[dataframe['Season'] == year]
    data = data[data['Name'] == name_components[2]]
    data = data[data['TimeStamp'] == name_components[3] + name_components[4].split('.')[0]] 
    latitude = data.iloc[0]['Latitude']
    longitude = data.iloc[0]['Longitude']

    # Write location of tropical cyclone into a file
    with open(IMG_DIR + '/' + item.split('.')[0] + '.location.tsv','w') as f:
        f.writelines([str(latitude) +','+ str(longitude)])

# print(readings)

# Timestamps from download links
#     lc = [l.split('/') for l in links_by_year[year]]
#     available_timestamps = [l[10].split('.')[0] + l[10].split('.')[1] for l in lc]

#     #Extract good download links for each cyclone 
#     for cyclone in cyclone_with_track:
#         # Timestamp from IBTrACS
#         timestamps = storm_data[storm_data['Name']==cyclone]['ISO_time'].tolist()
#         c_timestamps = [time.strftime("%Y%m%d%H%M",datetime.datetime.strptime(s, "%Y-%m-%d %H:%M:%S").timetuple()) for s in timestamps]


# # Filter out only cyclones after 2001
# data = data[data['Season'] >= 2001]

# # Get unique cyclone name from ibtracs
# storm_names = data.groupby('Season')['Name'].apply(np.unique)

# # Read in all download links
# with open('tutorial/cyclone.html', 'r') as f:
#     links = f.readlines()

# # Split download links into a dictionary with year as key
# links_by_year = {}
# for link in links:
#     year = int('20' + link.split('/')[4][2:])
#     if year not in links_by_year.keys():
#         links_by_year[year] = [link]
#     else:
#         links_by_year[year].append(link)

# # Extract good download links
# good_dl = []
# total_of_good_cyclones = 0

# for year in storm_names.index:
#     # Cyclone name in IBTrACS
#     ib_names = storm_names[year]
#     # Cyclone name in download links
#     dl_names = [c.split('/')[6].split('.')[1] for c in links_by_year[year]]

#     # Extract cyclone with records in IBTrACS
#     cyclone_with_track = []
#     for cyclone in ib_names:
#         if cyclone in dl_names:
#             cyclone_with_track.append(cyclone)
    
#     # Get all storm data for the year in IBTrACS
#     storm_data = data[data['Season'] == year]

#     # Timestamps from download links
#     lc = [l.split('/') for l in links_by_year[year]]
#     available_timestamps = [l[10].split('.')[0] + l[10].split('.')[1] for l in lc]

#     #Extract good download links for each cyclone 
#     for cyclone in cyclone_with_track:
#         # Timestamp from IBTrACS
#         timestamps = storm_data[storm_data['Name']==cyclone]['ISO_time'].tolist()
#         c_timestamps = [time.strftime("%Y%m%d%H%M",datetime.datetime.strptime(s, "%Y-%m-%d %H:%M:%S").timetuple()) for s in timestamps]

#         good_index = []
#         i = 0
#         lc_len = len(lc)
#         while i < lc_len:
#             if dl_names[i] == cyclone and available_timestamps[i] in c_timestamps:
#                 good_index.append(i)
#             i+=1

#         for i in good_index:
#             good_dl.append(links_by_year[year][i])
#     print(year, len(cyclone_with_track))
#     total_of_good_cyclones += len(cyclone_with_track)


# print(len(good_dl))
# print(total_of_good_cyclones)

# with open('download.txt','w') as f:
#     for item in good_dl:
#         f.writelines('%s' % item)
# print(count)
# import cv2

# img = cv2.imread('images1/15_SHEM_JOALANE_20150403_0600.jpg')
# gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# edges = cv2.Canny(gray,150,200,apertureSize = 3)
# cv2.imwrite('tescanny.jpg',edges)

# lines = cv2.HoughLines(edges,1,np.pi/180,200)
# for rho,theta in lines[0]:
#     a = np.cos(theta)
#     b = np.sin(theta)
#     x0 = a*rho
#     y0 = b*rho
#     x1 = int(x0 + 1000*(-b))
#     y1 = int(y0 + 1000*(a))
#     x2 = int(x0 - 1000*(-b))
#     y2 = int(y0 - 1000*(a))
   
#     cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)

# cv2.imwrite('testhoughlines.jpg',img)

#files = os.listdir('test')

#for f in files:
# image = Image.open('images1/15_SHEM_JOALANE_20150403_0600.jpg')
# image = image.convert('RGB')
# pixels = np.array(image)
# print(pixels.shape)

# height, width, _  = pixels.shape

# # Bottom border crop    
# t_pixels = pixels[:954,:,:]

# t_image = Image.fromarray(t_pixels)
# t_image.show()

# h_grid = []

# # Get horizontal grid line
# r = 0
# while r < height:
#     # Get pixels from row
#     r_pixels = pixels[r,:,:]
#     i = 0
#     r_pixels_len = len(r_pixels)
#     equal = True
#     while i < r_pixels_len - 1:
#         if (r_pixels[i] != r_pixels[i+1]).any():
#             equal = False
#             break
#         i+=1
#     if equal:
#         h_grid.append(r)
#     r+=1 
# print(h_grid)



    # # Get columns not black
    # c = 0
    # left_bound = 0
    # right_bound = 0
    # found_left = False
    # while c < width:
    #     no_of_null_pixels = 0
    #     c_pixels = pixels[:,c,:]
        
    #     for p in c_pixels:
    #         if (p == null_pixel).all():
    #             no_of_null_pixels += 1
        
    #     ratio_of_null_pixels = no_of_null_pixels / len(c_pixels)
    #     # if c== 0:
    #     #     print(c_pixels)
    #     #     print(ratio_of_null_pixels)
    #     if found_left and ratio_of_null_pixels > 0.9:
    #         right_bound = c
    #         break 

    #     if not found_left:
    #         left_bound = c        
            
    #         # If have moved onto image region
    #         if ratio_of_null_pixels < 0.9:
    #             found_left = True
        
    #     c += 1 

    #     if c == width and right_bound == 0:
    #         right_bound = c

    # t_pixels = pixels[top_bound:bottom_bound,left_bound:right_bound,:]

    # t_image = Image.fromarray(t_pixels)
    # t_image.show()