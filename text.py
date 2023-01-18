import cv2 as cv
import pytesseract
import PIL.Image

"""
Python segmentation modes:
0 = Orientation and script detection (OSD) only.
1 = Automatic page segmentation with OSD.
2 = Automatic page segmentation, but no OSD, or OCR.
3 = Fully automatic page segmentation, but no OSD. (Default)
4 = Assume a single column of text of variable sizes.
5 = Assume a single uniform block of vertically aligned text.
6 = Assume a single uniform block of text.
7 = Treat the image as a single text line.
8 = Treat the image as a single word.
9 = Treat the image as a single word in a circle.
10 = Treat the image as a single character.
11 = Sparse text. Find as much text as possible in no particular order.
12 = Sparse text with OSD.
13 = Raw line. Treat the image as a single text line, bypassing hacks that are Tesseract-specific.

"""

"""
OCR Engine modes:
0 = Original Tesseract only.
1 = Neural nets LSTM only.
2 = Tesseract + LSTM.
3 = Default, based on what is available.
"""

myconfig = r'--oem 3 --psm 11'

wellPerformingConfigs = {
    1: r'--oem 3 --psm 11'
}

text_cropped = pytesseract.image_to_string(PIL.Image.open('cropped.png'), config=myconfig)
text_cropped_1 = pytesseract.image_to_string(PIL.Image.open('cropped2.png'), config=myconfig)
text_uncropped = pytesseract.image_to_string(PIL.Image.open('uncropped.1920x1280.png'), config=myconfig)
# print(text_cropped + "\n\n\n"+ text_cropped_1)
print(text_uncropped)