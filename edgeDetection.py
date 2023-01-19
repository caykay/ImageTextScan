import cv2
import numpy as np
import matplotlib.pyplot as plt

# IMAGE PATHS-uncropped
uncropped1 = "uncropped.1920x1080.png"
uncropped2 = "1280x720.png"
uncropped3 = "1440x900.png"
uncropped4 = "1680x1050.png"
uncropped5 = "uncropped_mobile.PNG"

img = cv2.imread(uncropped5)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# use canny algorithm or binary threshold to make things easier for the contour finder
# by finding edges for unique shapes in the image
# using canny has yielded great results so far
edges = cv2.Canny(gray, 50,200, apertureSize=3, L2gradient=True)

# apply binary thresholding
# ret, thresh = cv2.threshold(gray, 150, 200, cv2.THRESH_BINARY)

# find contours from the edge map
contours, hierarchy = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

## As contours aren't quite rectangles, we can't just use the area to find the largest one
## So we'll use the bounding box area instead and find the largest of those

# get contour with largest bounding box
def get_largest_contour(contours):
    """Returns the contour with the largest bounding box area"""
    max_area = 0;
    largest_contour = (None, 0)  # (contour, area)
    for c in contours:
        # get area of bounding box
        area = get_contour_bounding_area(c)
        largest_contour = max((c, area), largest_contour, key=lambda x: x[1])
    return largest_contour[0]

def get_contour_bounding_area(contour):
    """Returns the area of the bounding box of the contour"""
    x, y, w, h = cv2.boundingRect(contour)
    return w * h

def get_aspect_ratio(contour):
    """Returns the aspect ratio of the bounding box of the contour"""
    x, y, w, h = cv2.boundingRect(contour)
    return w / h

# sort contours by bounding box area
sorted_contours = sorted(contours, key=get_contour_bounding_area, reverse=True)

print("Largest Contour Aspect Ratio: ", get_aspect_ratio(sorted_contours[0]))
print(img.shape[1]/img.shape[0])
# get contour bounding box
x1, y1, w1, h1 = cv2.boundingRect(sorted_contours[0])
x2, y2, w2, h2 = cv2.boundingRect(sorted_contours[1])
cv2.rectangle(img, (x1, y1), (x1+w1, y1+h1), (255, 0, 0), 2)
cv2.rectangle(img, (x2, y2), (x2+w2, y2+h2), (255, 0, 0), 2)

# crop
# crop = img[y:y+h, x:x+w]

# display
_, axs = plt.subplots(2,1)
axs[0].imshow(img)
axs[1].imshow(edges, cmap='gray')
plt.show()

"""
1920x1080
img aspect-ratio: 1.7777777777777777
largest contour aspect-ratio: 0.5873959571938169

1680x1050
img aspect-ratio: 1.6
largest contour aspect-ratio: 0.5130641330166271

1440x900
img aspect-ratio: 1.6
largest contour aspect-ratio: 0.5145631067961165

1280x720
img aspect-ratio: 1.7777777777777777
largest contour aspect-ratio: 0.5871886120996441
"""

# https://learnopencv.com/contour-detection-using-opencv-python-c/