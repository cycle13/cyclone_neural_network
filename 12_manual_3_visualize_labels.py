from __future__ import print_function
try:
    # for Python2
    from Tkinter import *
except ImportError:
    # for Python3
    from tkinter import *
from PIL import ImageTk
from helpers import cv2DrawText
from helpers import *


imgDir = "images1"

#no need to change these
drawingImgSize = 1000
boxWidth = 10
boxHeight = 2


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



def buttonPressedCallback(s):
    global global_lastButtonPressed
    global_lastButtonPressed = s

# create UI
objectNames = ["SHOW NEXT"]
tk = Tk()
w = Canvas(tk, width=len(objectNames) * boxWidth, height=len(objectNames) * boxHeight, bd = boxWidth, bg = 'white')
w.grid(row = len(objectNames), column = 0, columnspan = 2)
for objectIndex,objectName in enumerate(objectNames):
    b = Button(width=boxWidth, height=boxHeight, text=objectName, command=lambda s = objectName: buttonPressedCallback(s))
    b.grid(row = objectIndex, column = 0)

# Loop over all images
imgFilenames = getFilesInDirectory(imgDir, ".jpg")

for imgIndex, imgFilename in enumerate(imgFilenames):
    print (imgIndex, imgFilename)
    
    # Load image, box, labels and actual location
    img = imread(os.path.join(imgDir,imgFilename))
    boxPath = os.path.join(imgDir, imgFilename[:-4] + ".bboxes.tsv")
    labelsPath = os.path.join(imgDir, imgFilename[:-4] + ".bboxes.labels.tsv")
    locationPath = os.path.join(imgDir, imgFilename[:-4] + ".location.tsv")
    box = [ToIntegers(rect) for rect in readTable(boxPath)][0]
    with open(labelsPath,'r') as f:
        labels = f.readlines()[0].strip()

    labels = labels[1:-1].split(',')
    top = labels[0].split("'")[1]
    bottom = labels[1].split("'")[1]
    left = labels[2].split("'")[1]
    right = labels[3].split("'")[1]

    with open(locationPath, 'r') as f:
        location = f.readlines()[0].split(',')
    
    latitude = float(location[0])
    longitude = float(location[1])
    print(latitude, longitude)
    bottom_t = coord_transform(bottom)
    top_t = coord_transform(top)
    left_t = coord_transform(left)
    right_t = coord_transform(right)

    lat = box[3] + (latitude - bottom_t) * (box[1]-box[3])/(top_t - bottom_t)
    
    lon_dif = right_t - left_t
    if lon_dif < 0:
        right_t += 360
    lon_dif = longitude - left_t
    if lon_dif < 0:
        longitude += 360
    lon = box[0] + (longitude - left_t) * (box[2]-box[0])/(right_t - left_t)
    print(lat, lon)
    imgCopy = img.copy()

    # Annotate the bounding box
    color = (255, 0, 0)
    
    drawRectangles(imgCopy, [box], color = color, thickness = 2)

    drawRectangles(imgCopy, 
        [
            [box[0],box[1]-23,box[0]+getDrawTextWidth(top),box[1]], 
            [box[2]-getDrawTextWidth(bottom),box[3]-23,box[2],box[3]],
            [box[0],box[1]+50,box[0]+getDrawTextWidth(left),box[1]+78],
            [box[2]-getDrawTextWidth(right),box[1]+50,box[2],box[1]+78],
        ], color = color, thickness = -1)
    
    drawRectangles(imgCopy,[[lon-1,lat-1,lon+1,lat+1]], color=color, thickness=1)

    cv2DrawText(imgCopy, (box[0]+3,box[1]-7), top, color = (255,255,255), colorBackground=color)
    cv2DrawText(imgCopy, (box[2]-getDrawTextWidth(bottom)+3,box[3]-7), bottom, color = (255,255,255), colorBackground=color)
    cv2DrawText(imgCopy, (box[0]+3,box[1]+70), left, color = (255,255,255), colorBackground=color)
    cv2DrawText(imgCopy, (box[2]-getDrawTextWidth(right)+3,box[1]+70), right, color = (255,255,255), colorBackground=color)


    imgTk, _ = imresizeMaxDim(imgCopy, drawingImgSize, boUpscale = True)
    imgTk = ImageTk.PhotoImage(imconvertCv2Pil(imgTk))
    label = Label(tk, image=imgTk)
    label.grid(row=0, column=1, rowspan=drawingImgSize)

    # draw image in tk window
    imgTk, _ = imresizeMaxDim(imgCopy, drawingImgSize, boUpscale = True)
    imgTk = ImageTk.PhotoImage(imconvertCv2Pil(imgTk))
    label = Label(tk, image=imgTk)
    label.grid(row=0, column=1, rowspan=drawingImgSize)

    tk.update_idletasks()
    tk.update()
    ##writeFile(labelsPath, labels)
    # busy-wait until button pressed
    global_lastButtonPressed = None
    while not global_lastButtonPressed:
        tk.update_idletasks()
        tk.update()

tk.destroy()
print ("DONE.")
