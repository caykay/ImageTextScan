import cv2 as cv
import matplotlib.pyplot as plt
from easyOCR import textReader

uncropped1 = "uncropped.1920x1080.png"
uncropped2 = "1280x720.png"
uncropped3 = "1440x900.png"
uncropped4 = "1680x1050.png"

# CROP TESTING
img1 = cv.imread(uncropped1)
img2 = cv.imread(uncropped4)
# print(img.shape)

# shape: [1080, 1920, 3]
# img[y1:y2, x1:x2]
# crop-dimensions: img[120:960, 1305:1800]

# shape: [1050, 1680, 3]
# img[y1:y2, x1:x2]
# crop-dimensions: img[105:940, 1145:1575]

croppedImg1 = img1[120:960, 1305:1800]
croppedImg2 = img2[105:940, 1145:1575]

# fig, axs = plt.subplots(1, 2)
# axs[0].imshow(croppedImg1)
# axs[1].imshow(croppedImg2)
# plt.show()

textReader(croppedImg2)