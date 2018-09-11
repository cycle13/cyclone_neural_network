import os
import cv2
from helpers import *
from tkinter import *
import numpy as np
from PIL import ImageTk
from helpers import *
import pytesseract



def get_position(line_positions):
    # Sort position of line
    y = list(np.sort(line_positions))

    # Approximate the exact location of the lines
    def reduce_size(a):
        if len(a) < 2:
            return a

        if a[1] - a[0] <= 2:
            b = a[2:]
            b.insert(0, int(round((a[1] + a[0])/2)))
            return reduce_size(b)

        back = reduce_size(a[1:])
        back.insert(0, a[0])
        return back

    y = reduce_size(y)

    # Identify spacing between lines
    diff = []

    i = len(y) - 1
    while i > 0:
        j = i -1
        while j >= 0:
            difference = y[i] - y[j]
            diff.append(difference)
            j -= 1
        i -= 1

    bin_count = np.bincount(diff)
    ii = np.nonzero(bin_count)[0]
    freq = np.vstack((ii,bin_count[ii])).T

    freq = freq[freq[:,1].argsort()]
    minimum = 1000

    i = freq.shape[0] -1
    while i >= 0:
        if freq[i][1] < 2 or freq[i][1] < freq[-1][1]:
            break

        dist = freq[i][0]
        if dist > 5 and dist < minimum:
            minimum = dist

        i -= 1

    # Identify position of grid line
    position = []
    if minimum != 1000:
        # Identify spacing between lines
        i = len(y)- 1
        while i > 0:
            j = i -1
            while j >= 0:
                if y[i] - y[j] == minimum:
                    if y[i] not in position:
                        position.append(int(y[i]))
                    position.append(int(y[j]))
                    i = j + 1
                j -= 1
            i -= 1
    # print(freq)
    return position, minimum




def get_coordinate(image_file):
    TOP = 50
    BOTTOM = 954
    LEFT = 60
    LEFT_B = 405
    RIGHT = 964


    # img = cv2.imread(image_file_location + '.jpg')
    # cv2.imwrite(OUTPUT_DIR + "/image_hough_original.jpg", img)
    # image_file = image_file[TOP:BOTTOM, LEFT:RIGHT]
    # image_file = image.copy()

    gray = cv2.cvtColor(image_file,cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray,50,150,apertureSize = 3)

    lines = cv2.HoughLines(edges,1,np.pi/180,500)

    y = []
    x = []
    for line in lines:

        for rho,theta in line:
            angle = theta / np.pi * 180

            if angle > 89.5 and angle < 90.5:
                b = np.sin(theta)
                y0 = b*rho
                if y0 < BOTTOM:
                    y.append(y0)

            if angle > 359.5 or angle < 0.5:
                a = np.cos(theta)
                x0 = a*rho
                x.append(x0)


    x, diff_x = get_position(x)
    x = np.sort(x)
    y, diff_y = get_position(y)
    y = np.sort(y)

    if len(x) == 0 or len(y) == 0:
        return [], [], x, y, diff_x, diff_y




    # Detecting latitude leftside
    lat_coords = {}
    left_coord = x[0]

    if y[0] - TOP <= diff_y and y[0] - TOP > 0:
        img_temp = image_file[TOP:y[0]+1, :left_coord+1]
        img_temp = image_file[y[0]-30:y[0]+1, 0:left_coord+1]
        
        img_temp = cv2.cvtColor(img_temp, cv2.COLOR_BGR2GRAY)
        img_temp = cv2.threshold(img_temp, 0, 255,
                            cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        
        recognised_character = pytesseract.image_to_string(img_temp)
        if recognised_character and \
                len(recognised_character) in [2,3] and \
                recognised_character[:-1].isdigit() and \
                recognised_character[-1] in ['N', 'S']:
            lat_coords[y[0]] = recognised_character

    i = 1
    while i < len(y):
        img_temp = image_file[y[i-1]:y[i]+1, :left_coord+1]
        img_temp = image_file[y[i]-30:y[i]+1, :left_coord+1]

        img_temp = cv2.cvtColor(img_temp, cv2.COLOR_BGR2GRAY)
        img_temp = cv2.threshold(img_temp, 0, 255,
                        cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

        recognised_character = pytesseract.image_to_string(img_temp)
        if recognised_character and \
                len(recognised_character) in [2,3] and \
                recognised_character[:-1].isdigit() and \
                recognised_character[-1] in ['N', 'S']:
            lat_coords[y[i]] = recognised_character
        i+=1

    # Detecting latitude rightside
    right_coords = LEFT_B if LEFT_B > x[-1] else x[-1]
    if y[0] - TOP <= diff_y and y[0] - TOP > 0:
        img_temp = image_file[TOP:y[0]+1, right_coords:]
        img_temp = image_file[y[0]-30:y[0]+1, right_coords:]

        img_temp = cv2.cvtColor(img_temp, cv2.COLOR_BGR2GRAY)
        img_temp = cv2.threshold(img_temp, 0, 255,
                            cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

        recognised_character = pytesseract.image_to_string(img_temp)
        if recognised_character and \
                len(recognised_character) in [2,3] and \
                recognised_character[:-1].isdigit() and \
                recognised_character[-1] in ['N', 'S']:
            lat_coords[y[0]] = recognised_character

    i = 1
    while i < len(y):
        img_temp = image_file[y[i-1]:y[i]+1, right_coords:]
        img_temp = image_file[y[i]-30:y[i]+1, right_coords:]

        img_temp = cv2.cvtColor(img_temp, cv2.COLOR_BGR2GRAY)
        img_temp = cv2.threshold(img_temp, 0, 255,
                            cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

        recognised_character = pytesseract.image_to_string(img_temp)
        if recognised_character and \
                len(recognised_character) in [2,3] and \
                recognised_character[:-1].isdigit() and \
                recognised_character[-1] in ['N', 'S']:
            lat_coords[y[i]] = recognised_character
        i+=1


    # Detecting longitude topside
    lon_coords = {}
    top_coord = y[0]

    if image_file.shape[1] - x[-1] <= diff_x:
        img_temp = image_file[:top_coord+1, x[-1]:]
        img_temp = np.rot90(image_file[:70,x[-1]:x[-1]+30],3)

        img_temp = cv2.cvtColor(img_temp, cv2.COLOR_BGR2GRAY)
        img_temp = cv2.threshold(img_temp, 0, 255,
                            cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

        recognised_character = pytesseract.image_to_string(img_temp)
        if recognised_character and \
                len(recognised_character) in [2,3] and \
                recognised_character[:-1].isdigit() and \
                recognised_character[-1] in ['W', 'E']:
            lon_coords[x[-1]] = recognised_character

    i = len(x) - 2
    while i >= 0 and x[i] >= LEFT_B:
        img_temp = np.rot90(image_file[:top_coord+1,x[i]:x[i+1]+1],3)
        img_temp = np.rot90(image_file[:70,x[i]-1:x[i]+30],3)

        img_temp = cv2.cvtColor(img_temp, cv2.COLOR_BGR2GRAY)
        img_temp = cv2.threshold(img_temp, 0, 255,
                            cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

        recognised_character = pytesseract.image_to_string(img_temp)
        if recognised_character and \
                len(recognised_character) in [2,3,4] and \
                recognised_character[:-1].isdigit() and \
                recognised_character[-1] in ['W', 'E']:
            lon_coords[x[i]] = recognised_character
        i-=1
    print(lat_coords)
    print(lon_coords)

    return lat_coords, lon_coords, x, y, diff_x, diff_y
# cv2.imwrite(OUTPUT_DIR + '/image_hough1.jpg',img)
