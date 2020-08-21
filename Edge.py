import imageio as iio
import matplotlib.pyplot as plt
import cv2
import numpy as np

# Read image
img = cv2.imread("cat.jpg")
# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# Threshold greyscale image to binary
_, thres = cv2.threshold(gray, 253, 255, cv2.THRESH_BINARY_INV)

# Square image kernel used for erosion
kernel = np.ones((5,5), np.uint8)
# Open - meaning to erode and dilate (gets rid of the fuzzy outlines)
thres_open = cv2.morphologyEx(thres, cv2.MORPH_OPEN, kernel)
# Close - meaning to get rid of irregularities within shape - e.g. whiskers
thres_close = cv2.morphologyEx(thres_open, cv2.MORPH_CLOSE, kernel)

# 1st method - Edge detection
edges = cv2.Canny(thres_close, 200, 300)
# 2nd method - Contours
contours, _ = cv2.findContours(thres_close, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contoured_image = cv2.drawContours(np.ones(gray.shape), contours, -1, (255, 0, 0), thickness=4)

# Plots
plt.subplot(231)
plt.imshow(gray, cmap="gray")
plt.title('Greyscale'), plt.xticks([]), plt.yticks([])

plt.subplot(232)
plt.imshow(thres, cmap="gray")
plt.title('Thresholded'), plt.xticks([]), plt.yticks([])

plt.subplot(233)
plt.imshow(thres_open, cmap="gray")
plt.title('Thresholded and "opened"'), plt.xticks([]), plt.yticks([])

plt.subplot(234)
plt.imshow(thres_close, cmap="gray")
plt.title('Thresholded and "closed"'), plt.xticks([]), plt.yticks([])

plt.subplot(235)
plt.imshow(edges, cmap="gray")
plt.title('Canny'), plt.xticks([]), plt.yticks([])

plt.subplot(236)
plt.imshow(contoured_image, cmap="gray")
plt.title('Contour'), plt.xticks([]), plt.yticks([])

plt.show()
