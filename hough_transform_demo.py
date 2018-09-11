import cv2
import numpy as np
import math

default_file =  "demo/02_ATL_BERTHA_20020808_1200.jpg"
filename = default_file
# Loads an image
src = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
# Check if image is loaded fine


# Copy edges to the images that will display the results in BGR
cdst = cv.cvtColor(dst, cv.COLOR_GRAY2BGR)
cdstP = np.copy(cdst)

#  Standard Hough Line Transform
lines = cv2.HoughLines(dst, 1, np.pi / 180, 150, None, 0, 0)

 # Draw the lines
if lines is not None:
    for i in range(0, len(lines)):
        rho = lines[i][0][0]
        theta = lines[i][0][1]
        a = math.cos(theta)
        b = math.sin(theta)
        x0 = a * rho
        y0 = b * rho
        pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
        pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
        cv2.line(dst, pt1, pt2, (0,0,255), 3, cv2.LINE_AA)

# Show results
cv2.imshow("Source", src)
cv2.imshow("Detected Lines (in red) - Standard Hough Line Transform", dst)
# cv2.imshow("Detected Lines (in red) - Probabilistic Line Transform", cdstP)

# Wait and Exit
cv2.waitKey()