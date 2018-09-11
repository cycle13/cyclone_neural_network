from __future__ import print_function
try:
    # for Python2
    from Tkinter import *
except ImportError:
    # for Python3
    from tkinter import *
from PIL import ImageTk
from helpers import *



# PARAMETERS
imgDir = "images1"
drawingImgSize = 1000
boxWidth = 100
boxHeight = 2



# MAIN

# Define callback function for Annotate button
def submitLabels(s):
    global global_coordinates
    global_coordinates = [e1.get(),e2.get(),e3.get(),e4.get()]

# Creating UI
tk = Tk()
w = Canvas(tk, width=2*boxWidth, height= 4 * boxHeight, bd = boxWidth, bg = 'white')

w.grid(row = 5, column = 0, columnspan = 3)

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
Button(tk, text='Annotate',
    command=lambda s = []: submitLabels(s)).grid(row = 4)



# loop over all images
imgFilenames = getFilesInDirectory(imgDir, ".jpg")
imgFilenames += getFilesInDirectory(imgDir, ".png")
for imgIndex, imgFilename in enumerate(imgFilenames):
    print (imgIndex, imgFilename)
    labelsPath = os.path.join(imgDir, imgFilename[:-4] + ".bboxes.labels.tsv")
    if os.path.exists(labelsPath):
        print ("Skipping image {:3} ({}) since annotation file already exists: {}".format(imgIndex, imgFilename, labelsPath))
        continue

    # load image and ground truth rectangles
    
    rectsPath = os.path.join(imgDir, imgFilename[:-4] + ".bboxes.tsv")
    if os.path.exists(rectsPath):
       
        rects = [ToIntegers(rect) for rect in readTable(rectsPath)]

        img = imread(os.path.join(imgDir,imgFilename))
        
        # annotate each rectangle in turn
        labels = []
        for rectIndex,rect in enumerate(rects):
            imgCopy = img.copy()
            drawRectangles(imgCopy, [rect], thickness = 15)

            # draw image in tk window
            imgTk, _ = imresizeMaxDim(imgCopy, drawingImgSize, boUpscale = True)
            imgTk = ImageTk.PhotoImage(imconvertCv2Pil(imgTk))
            label = Label(tk, image=imgTk)
            label.grid(row=0, column=2, rowspan=drawingImgSize)
            tk.update_idletasks()
            tk.update()

            # busy-wait until button pressed
            global_coordinates = None
            while not global_coordinates:
                tk.update_idletasks()
                tk.update()

            # store result
            print ("Coordinates entered = ", global_coordinates)
            labels.append(global_coordinates)

        writeFile(labelsPath, labels)
tk.destroy()
print ("DONE.")
