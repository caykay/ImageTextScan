import cv2 as cv
import easyocr
from matplotlib import pyplot as plt
import numpy as np

# IMAGE PATHS
cropped = "cropped.png"
cropped1 = "cropped1.png"
cropped2 = "cropped2.png"
cropped3 = "Genshin Impact Screenshot 2023.01.18 - 22.51.16.62 (2).png"
cropped4 = "Genshin Impact Screenshot 2022.01.11 - 00.51.03.59 (2).png"
uncropped1 = "uncropped.1920x1080.png"
uncropped2 = "1280x720.png"
uncropped3 = "1440x900.png"
uncropped4 = "1680x1050.png"

imgLocation = cropped4

# READ IMAGE
reader = easyocr.Reader(['en'], gpu=True)
result = reader.readtext(imgLocation)
# for line in result:
#     print(line)

# DISPLAY IMAGE
topLeft = tuple(result[0][0][0])
bottomRight = tuple(result[0][0][2])
text = result[0][1]
font = cv.FONT_HERSHEY_SIMPLEX
# visualize
img = cv.imread(imgLocation)
memo = []
for detection in result:
    topLeft = tuple([int(var) for var in detection[0][0]])
    bottomRight = tuple([int(var) for var in detection[0][2]])
    text = detection[1]
    memo.append(text)
    img = cv.rectangle(img, topLeft, bottomRight, (0, 255, 0), 2)
    img = cv.putText(img, text, topLeft, font, 0.7, (255, 0, 0), 1, cv.LINE_AA)

print(memo)
plt.imshow(img)
plt.show()

def textReader(img):
    result = reader.readtext(img)
    # DISPLAY IMAGE
    topLeft = tuple(result[0][0][0])
    bottomRight = tuple(result[0][0][2])
    text = result[0][1]
    font = cv.FONT_HERSHEY_SIMPLEX
    # visualize
    memo = []
    for detection in result:
        topLeft = tuple([int(var) for var in detection[0][0]])
        bottomRight = tuple([int(var) for var in detection[0][2]])
        text = detection[1]
        memo.append(text)
        img = cv.rectangle(img, topLeft, bottomRight, (0, 255, 0), 2)
        img = cv.putText(img, text, topLeft, font, 0.7, (255, 0, 0), 1, cv.LINE_AA)

    print(memo)
    plt.imshow(img)
    plt.show()

