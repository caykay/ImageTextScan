import cv2 as cv
import easyocr
from matplotlib import pyplot as plt
import numpy as np

# IMAGE PATHS
cropped = "cropped.png"
cropped1 = "cropped1.png"
cropped2 = "cropped2.png"
uncropped1 = "uncropped.1920x1080.png"
uncropped2 = "1280x720.png"
uncropped3 = "1440x900.png"
uncropped4 = "1680x1050.png"

# READ IMAGE
reader = easyocr.Reader(['en'], gpu=True)
result = reader.readtext(cropped)
for line in result:
    print(line)
