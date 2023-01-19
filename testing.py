import cv2 as cv
import matplotlib.pyplot as plt

uncropped1 = "uncropped.1920x1080.png"
uncropped2 = "1280x720.png"
uncropped3 = "1440x900.png"
uncropped4 = "1680x1050.png"

# CROP TESTING
img = cv.imread(uncropped4)
print(img.shape)

# shape: [1050, 1680, 3]
# start: (1140, 100)
# end: (1570, 940)

croppedImg = img[105:940, 1145:1575] # img[y1:y2, x1:x2]

plt.imshow(croppedImg)
plt.show()