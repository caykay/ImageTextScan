import cv2 as cv
import pytesseract
import PIL.Image
from matplotlib import pyplot as plt

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

myconfig = r'--oem 3 --psm 6'

wellPerformingConfigs = {
    1: r'--oem 3 --psm 11',
    2: r'--oem 3 --psm 3',
    3: r'--oem 3 --psm 1',
}

# text_cropped = pytesseract.image_to_string(PIL.Image.open('cropped.png'), config=myconfig)
# text_cropped_1 = pytesseract.image_to_string(PIL.Image.open('cropped2.png'), config=myconfig)
# text_uncropped = pytesseract.image_to_string(PIL.Image.open('uncropped.1920x1080.png'), config=myconfig)
# # print(text_cropped + "\n\n\n"+ text_cropped_1)
# print(text_uncropped)

# text1 = pytesseract.image_to_string(PIL.Image.open('1280x720.png'), config=myconfig)
# text2 = pytesseract.image_to_string(PIL.Image.open('1440x900.png'), config=myconfig)
# text3 = pytesseract.image_to_string(PIL.Image.open('1680x1050.png'), config=myconfig)

# print(text2)
cropped = "cropped.png"
cropped1 = "cropped1.png"
cropped2 = "cropped2.png"
uncropped1 = "uncropped.1920x1080.png"
uncropped2 = "1280x720.png"
uncropped3 = "1440x900.png"
uncropped4 = "1680x1050.png"

img = cv.imread(uncropped4)
# convert to grayscale
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# image binarization
_,tresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY|cv.THRESH_OTSU)
# height, width = img.shape
# print(width, height)

## draw boxes on individual characters
# boxes = pytesseract.image_to_boxes(img, config=myconfig)
# print(boxes)
# for box in boxes.splitlines():
#     box = box.split(' ')
#     x, y, w, h = int(box[1]), int(box[2]), int(box[3]), int(box[4])
#     cv.rectangle(img, (x, height-y), (w, height-h), (255, 0, 0), 1)
#     cv.putText(img, box[0], (x, height-y+25), cv.FONT_HERSHEY_COMPLEX, 1, (50, 50, 255), 2)

# draw boxes on words
data = pytesseract.image_to_data(tresh, config=myconfig, output_type=pytesseract.Output.DICT, lang='eng')
print(data['text'])
amountOfWords = len(data['text'])
for i in range(amountOfWords):
    if float(data['conf'][i]) > 30:
        x, y, w, h = int(data['left'][i]), int(data['top'][i]), int(data['width'][i]), int(data['height'][i])
        cv.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 1)
        cv.putText(img, data['text'][i], (x, y), cv.FONT_HERSHEY_COMPLEX, 1, (50, 50, 255), 2)


plt.imshow(tresh, cmap='gray')
plt.show()