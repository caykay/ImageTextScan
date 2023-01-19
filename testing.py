import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import pytesseract
import PIL.Image

uncropped1 = "uncropped.1920x1080.png"
uncropped2 = "1280x720.png"
uncropped3 = "1440x900.png"
uncropped4 = "1680x1050.png"
uncropped5 = "uncropped_mobile.PNG"

def get_text_bounds(filename):
    img = cv.imread(filename)
    height, width = img.shape[:2]

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    _,thresh_dark = cv.threshold(gray, 0, 255, cv.THRESH_OTSU|cv.THRESH_BINARY_INV)
    _,thresh_light = cv.threshold(gray, 0, 255, cv.THRESH_OTSU|cv.THRESH_BINARY)

    rect_kernel = cv.getStructuringElement(cv.MORPH_RECT, (15, 10))
    dilation_light = cv.dilate(thresh_light, rect_kernel, iterations = 1)
    dilation_dark = cv.dilate(thresh_dark, rect_kernel, iterations = 1)

    contours, hierarchy = cv.findContours(dilation_dark, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    contours = (*contours,*cv.findContours(dilation_light, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)[0])

    # set offsets for cutting out contours that close to the edge of the image
    width_offest, height_offset = 0.05 * width, 0.075 * height

    elligible_contours = []
    # draw contours for dark text and light text
    for cnt in contours:
        x, y, w, h = cv.boundingRect(cnt)
        # a bit of pruning to remove small contours and contours that are too close to the edge
        # as well as contours that are within 2/3 of the img width from the left
        if w < 10 or h < 10 \
            or x < (width/3 * 2) or x > (width - width_offest) \
            or y < height_offset or y > (height - height_offset) \
            or (x+w) > (width - width_offest) \
            or (y+h) > (height - height_offset)\
            or cnt.size == 0:
            continue
        elligible_contours.append(cnt)
        

    # draw contours for dark text and light text
    for cnt in elligible_contours:
        if cnt.size == 0:
            continue
        x, y, w, h = cv.boundingRect(cnt)
        cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    return img, elligible_contours

igm1, contours1 = get_text_bounds(uncropped1)
igm2, contours2 = get_text_bounds(uncropped2)
igm3, contours3 = get_text_bounds(uncropped3)
igm4, contours4 = get_text_bounds(uncropped4)
igm5, contours5 = get_text_bounds(uncropped5)

_, ax = plt.subplots(2, 3, figsize=(20, 10))
ax[0, 0].imshow(igm1)
ax[0, 1].imshow(igm2)
ax[0, 2].imshow(igm3)
ax[1, 0].imshow(igm4)
ax[1, 1].imshow(igm5)

plt.show()

textReader(croppedImg2)