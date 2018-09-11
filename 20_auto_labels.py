import os
import cv2
from helpers import *
from tkinter import *
import numpy as np
from PIL import ImageTk
from helpers import *
from hough_transform import get_coordinate
import pandas
import time
import datetime



# PARAMETERS
IMG_DIR = './demo'
# OUTPUT_DIR = './illustrations'
TOP = 50
BOTTOM = 954
LEFT = 60
LEFT_B = 405
RIGHT = 964
IMG_NO = 100
drawingImgSize = 800
boxWidth = 100
boxHeight = 2
color = (255, 0, 0)



def str_to_numeric_coords(data):
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



def numeric_to_str_coords(data, direction):
    data = int(round(data))
    if direction == 'lon':
        if data < 0:
            return str(-data) + 'W'
        else:
            return str(data) + 'E'

    if direction == 'lat':
        if data < 0:
            return str(-data) + 'S'
        else:
            return str(data) + 'N'

left = right = top = bottom = lat =  lon = label = submitted = skipped = labelled = top_edge = bottom_edge = left_edge = right_edge = imgCopy = None

def get_center_of_cyclone(top, bottom, left, right, latitude, longitude):
    global top_edge, bottom_edge, left_edge, right_edge
    bottom_t = str_to_numeric_coords(bottom)
    top_t = str_to_numeric_coords(top)
    left_t = str_to_numeric_coords(left)
    right_t = str_to_numeric_coords(right)

    lat = bottom_edge - (latitude - bottom_t) * (bottom_edge-top_edge)/(top_t - bottom_t)

    lon_dif = right_t - left_t
    if lon_dif < 0:
        right_t += 360
    lon_dif = longitude - left_t
    if lon_dif < 0:
        longitude += 360

    lon = left_edge + (longitude - left_t) * (right_edge-left_edge)/(right_t - left_t)
    print(lon)
    return lat, lon

def label_bbox(top, bottom, left, right):
    global top_edge, bottom_edge, left_edge, right_edge, imgCopy

    drawRectangles(imgCopy,
        [[left_edge,top_edge-23,left_edge+getDrawTextWidth(top),top_edge],
            [right_edge-getDrawTextWidth(bottom),bottom_edge-23,right_edge,bottom_edge],
            [left_edge,top_edge+50,left_edge+getDrawTextWidth(left),top_edge+78],
            [right_edge-getDrawTextWidth(right),top_edge+50,right_edge,top_edge+78],
    ], color = color, thickness = -1)

    cv2DrawText(imgCopy, (left_edge+3,top_edge-7), top, color = (255,255,255), colorBackground=color)
    cv2DrawText(imgCopy, (right_edge-getDrawTextWidth(bottom)+3,bottom_edge-7), bottom, color = (255,255,255), colorBackground=color)
    cv2DrawText(imgCopy, (left_edge+3,top_edge+70), left, color = (255,255,255), colorBackground=color)
    cv2DrawText(imgCopy, (right_edge-getDrawTextWidth(right)+3,top_edge+70), right, color = (255,255,255), colorBackground=color)



# Define callback function for Annotate button
def addOrAmend(s):
    global left, right, top, bottom, top_edge, bottom_edge, left_edge, right_edge, imgCopy, labelled, label, lat, lon

    top = e1.get()
    bottom = e2.get()
    left = e3.get()
    right = e4.get()

    label_bbox(top, bottom, left, right)

    print('Added coordinates')

    if item + '.wind.tsv' in files:
        name_comps = item.split('_')
        info = data[(data['Season']==int('20'+name_comps[0])) & (data['Name']==name_comps[2]) &
                        (data['ISO_time']==str(datetime.datetime.strptime(name_comps[3] + name_comps[4], "%Y%m%d%H%M")))]
        latitude = info['Latitude'].values[0]
        longitude = info['Longitude'].values[0]
        print(latitude, longitude)

        lat, lon = get_center_of_cyclone(top, bottom, left, right, latitude, longitude)

        drawRectangles(imgCopy, [[lon-1,lat-1,lon+1,lat+1]],  color = color, thickness = 2)

    imgTk, _ = imresizeMaxDim(imgCopy, drawingImgSize, boUpscale = True)
    imgTk = ImageTk.PhotoImage(imconvertCv2Pil(imgTk))
    label.configure(image=imgTk)
    label.image = imgTk
    labelled = True

def skip(s):
    global skipped
    skipped = True

def submit(s):
    global submitted
    submitted = True

# Creating UI
tk = Tk()
w = Canvas(tk, width=2*boxWidth, height= 4 * boxHeight, bd = boxWidth, bg = 'white')

w.grid(row = 7, column = 0, columnspan = 3)

Label(tk, text="Top").grid(row=0)
Label(tk, text="Bottom").grid(row=1)
Label(tk, text="Left").grid(row=2)
Label(tk, text="Right").grid(row=3)
e1 = Entry(tk)
e2 = Entry(tk)
e3 = Entry(tk)
e4 = Entry(tk)
e1.grid(row=0, column=1)
e2.grid(row=1, column=1)
e3.grid(row=2, column=1)
e4.grid(row=3, column=1)
Button(tk, text='Add/amend coordinates',
    command=lambda s = []: addOrAmend(s)).grid(row = 4)
Button(tk, text='Submit',
    command=lambda s = []: submit(s)).grid(row = 5)
Button(tk, text='Pass',
    command=lambda s = []: skip(s)).grid(row = 6)

# Extract centre of cyclone information
data = pandas.read_csv('ibtracs.csv', low_memory=False, header=1, skiprows=[2])
# Filter out only cyclones after 2001
data = data[data['Season'] >= 2001]
# # Get unique cyclone name from ibtracs
# storm_names = data.groupby('Season')['Name'].apply(np.unique)


# Main
files = os.listdir(IMG_DIR)

images = []

for item in files:
    if item.endswith('.jpg'):
        images.append(item)

image_files = [item.split('.')[0] for item in images]

for item in image_files:
    print(item)

    # Skipping images already with annotation
    if item + '.bboxes.tsv' in files:
        print('SKIPPED: Location already identified')
        continue

    # Read image and identify lines in the image
    img = cv2.imread(IMG_DIR + '/' + item + '.jpg')
    lat_coords, lon_coords, lons, lats, lon_diff, lat_diff = get_coordinate(img)

    print('Longitude positions: ', lons)
    print('Latitude positions: ', lats)

    # Draw the detected lines on the image
    for i in lons:
        x1 = x2 = i
        y1 = 0
        y2 = 1000
        cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)

    for i in lats:
        y1 = y2 = i
        x1 = 0
        x2 = 1000
        cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)

    # Skip if no line is detected
    if len(lons) < 2  or len(lats) < 2:
        print('SKIPPED: No line detected')
        continue

    # Else, draw biggest bounding box detected
    left_edge = np.min(lons)
    right_edge = np.max(lons)
    top_edge = np.min(lats)
    bottom_edge = np.max(lats)
    imgCopy = img.copy()
    drawRectangles(imgCopy, [[left_edge,top_edge,right_edge,bottom_edge]],  color = color, thickness = 2)

    if len(lon_coords) == 0  or len(lat_coords) == 0:
        print('Not able to read any position')

    else:
        # Annotate the bounding box if coordinates can be interpolated

        # Interpolate longitude 2 degrees each box
        # Find the closest position for interpolation
        spacing = 2/lon_diff
        all_lon_positions = np.array([*lon_coords.keys()])
        difference = np.abs(all_lon_positions - right_edge * np.ones(all_lon_positions.shape))
        lon_detected_position = all_lon_positions[np.argmin(difference)]
        lon_detected_coord = str_to_numeric_coords(lon_coords[lon_detected_position])
        r_position = lon_detected_coord + (right_edge - lon_detected_position) * spacing
        right = numeric_to_str_coords(r_position - 360, 'lon') if r_position > 180 else numeric_to_str_coords(r_position, 'lon')

        difference = np.abs(all_lon_positions - left_edge * np.ones(all_lon_positions.shape))
        lon_detected_position = all_lon_positions[np.argmin(difference)]
        lon_detected_coord = str_to_numeric_coords(lon_coords[lon_detected_position])
        l_position = lon_detected_coord + (left_edge - lon_detected_position) * spacing
        left = numeric_to_str_coords(l_position - 360, 'lon') if r_position > 180 else numeric_to_str_coords(l_position, 'lon')


        # Interpolate latitude 2 degrees each box
        # Find the closest position for interpolation
        spacing = 2/lat_diff
        all_lat_positions = np.array([*lat_coords.keys()])
        difference = np.abs(all_lat_positions - top_edge * np.ones(all_lat_positions.shape))
        lat_detected_position = all_lat_positions[np.argmin(difference)]
        lat_detected_coord = str_to_numeric_coords(lat_coords[lat_detected_position])
        t_position = lat_detected_coord - (top_edge - lat_detected_position) * spacing
        top = numeric_to_str_coords(t_position, 'lat')

        difference = np.abs(all_lat_positions - bottom_edge * np.ones(all_lat_positions.shape))
        lat_detected_position = all_lat_positions[np.argmin(difference)]
        lat_detected_coord = str_to_numeric_coords(lat_coords[lat_detected_position])
        b_position = lat_detected_coord - (bottom_edge - lat_detected_position) * spacing
        bottom = numeric_to_str_coords(b_position, 'lat')

        label_bbox(top, bottom, left, right)

        e1.delete(0,END)
        e1.insert(0,top)
        e2.delete(0,END)
        e2.insert(0,bottom)
        e3.delete(0,END)
        e3.insert(0,left)
        e4.delete(0,END)
        e4.insert(0,right)

        # Interpolate centre of cyclone if there is information
        if item + '.wind.tsv' in files:
            name_comps = item.split('_')
            info = data[(data['Season']==int('20'+name_comps[0])) & (data['Name']==name_comps[2]) &
                            (data['ISO_time']==str(datetime.datetime.strptime(name_comps[3] + name_comps[4], "%Y%m%d%H%M")))]
            latitude = info['Latitude'].values[0]
            longitude = info['Longitude'].values[0]
            print(latitude, longitude)

            lat, lon = get_center_of_cyclone(top, bottom, left, right, latitude, longitude)

            drawRectangles(imgCopy, [[lon-1,lat-1,lon+1,lat+1]],  color = color, thickness = 2)
        labelled = True

    # draw image in tk window
    imgTk, _ = imresizeMaxDim(imgCopy, drawingImgSize, boUpscale = True)
    imgTk = ImageTk.PhotoImage(imconvertCv2Pil(imgTk))
    label = Label(tk, image=imgTk)
    label.grid(row=0, column=2, rowspan=drawingImgSize)

    while not labelled and not skipped:
        tk.update_idletasks()
        tk.update()

    if labelled:
        while not submitted and not skipped:
            tk.update_idletasks()
            tk.update()

    if submitted:
        with open(IMG_DIR + '/' + item + '.bboxes.tsv', 'w') as f:
            f.write('{}\t{}\t{}\t{}\n'.format(left_edge, top_edge, right_edge, bottom_edge))
            # 44	177	1018	845
        with open(IMG_DIR + '/' + item + '.bboxes.labels.tsv', 'w') as f:
            f.write("['{}', '{}', '{}', '{}']\n".format(top, bottom, left, right))
            # ['32N', '26N', '100W', '90W']
        if item + '.wind.tsv' in files:
            with open(IMG_DIR + '/' + item + '.location.tsv', 'w') as f:
                f.write('{} {}\n'.format(int(round(lat)), int(round(lon))))

        print('DONE: Metadata saved')
    elif skipped:
        print('SKIPPED: Incomplete information')
    submitted = skipped = labelled = None

tk.destroy()
print ("DONE.")
