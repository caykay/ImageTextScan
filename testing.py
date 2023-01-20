import cv2 as cv
import easyocr
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
    """Gets the bounding boxes of the text in an image"""
    img = cv.imread(filename)
    height, width = img.shape[:2]

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    _,thresh_dark = cv.threshold(gray, 0, 255, cv.THRESH_OTSU|cv.THRESH_BINARY_INV)
    _,thresh_light = cv.threshold(gray, 0, 255, cv.THRESH_OTSU|cv.THRESH_BINARY)

    ## set kernel size for dilation
    rect_kernel_light = cv.getStructuringElement(cv.MORPH_RECT, (15, 10))
    rect_kernel_dark = cv.getStructuringElement(cv.MORPH_RECT, (15, 10))

    dilation_light = cv.dilate(thresh_light, rect_kernel_light, iterations = 1)
    dilation_dark = cv.dilate(thresh_dark, rect_kernel_dark, iterations = 1)

    contours, hierarchy = cv.findContours(dilation_dark, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    contours = (*contours,*cv.findContours(dilation_light, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)[0])

    ## set offsets for cutting out contours that close to the edge of the image
    width_offest, height_offset = 0.05 * width, 0.075 * height

    elligible_contours = []
    ## draw contours for dark text and light text
    for cnt in contours:
        x, y, w, h = cv.boundingRect(cnt)
        ## a bit of pruning to remove small contours and contours that are too close to the edge
        ## as well as contours that are within 2/3 of the img width from the left
        # TODO: prune contours based off of screen-size or test more screen sizes
        if w < 10 or h < 10 \
            or x < (width/3 * 2) or x > (width - width_offest) \
            or y < height_offset or y > (height - height_offset) \
            or (x+w) > (width - width_offest) \
            or (y+h) > (height - height_offset)\
            or cnt.size == 0:
            continue
        elligible_contours.append(cnt)
        
    ## re-order contours according to their y position
    elligible_contours = sorted(elligible_contours, key=lambda x: cv.boundingRect(x)[1])
    
    ## merge contours to form one large bounding rect
    top_left = [width, height]
    bottom_right = [0, 0]
    for cnt in elligible_contours:
        x, y, w, h = cv.boundingRect(cnt)
        top_left[0] = min(top_left[0], x)
        top_left[1] = min(top_left[1], y)
        bottom_right[0] = max(bottom_right[0], x+w)
        bottom_right[1] = max(bottom_right[1], y+h)

    text_area = (top_left[0], top_left[1], bottom_right[0], bottom_right[1])

    ## draw merged bounding rect
    img = cv.rectangle(img, (top_left[0], top_left[1]), (bottom_right[0], bottom_right[1]), (0, 0, 255), 2)
    return img, text_area, elligible_contours

def get_text_tessOCR(img_info):
    """Gets the text from the bounding boxes of the text in an image"""""
    
    def preprocess_text(text):
        """Preprocesses the text to remove newlines and spaces"""
        text = text.splitlines()
        return text

    img, bounding_area, contours = img_info
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    
    output = []
    config = r'--oem 3 --psm 6'

    ## process each contour (slower)
    for cnt in contours:
        x, y, w, h = cv.boundingRect(cnt)
        cropped_img = gray[y:y+h, x:x+w]
        text = pytesseract.image_to_string(cropped_img, config=config)
        img = cv.putText(img, text, (x, y), cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 1, cv.LINE_AA)
        output.append(text)
    print(output)

    ## process the merged bounding rect/contour (faster)
    # x, y, w, h = bounding_area
    # cropped_img = gray[y:y+h, x:x+w]
    # text = pytesseract.image_to_string(cropped_img, config=config)
    # text = preprocess_text(text)
    # print(text)

    # draw text on original image
    for cnt in contours:
        if cnt.size == 0:
            continue
        x, y, w, h = cv.boundingRect(cnt)
        img = cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    
    plt.imshow(img)
    plt.show()

def get_text_easyOCR(img_info):
    img, bounding_area, contours = img_info
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    x, y, w, h = bounding_area
    cropped_img = gray[y:y+h, x:x+w]
    # READ IMAGE
    reader = easyocr.Reader(['en'], gpu=True)
    result = reader.readtext(cropped_img)
    print([text_row[1] for text_row in result])

    # show text on original image
    for text_row in result:
        top_left = tuple([int(var) for var in text_row[0][0]])
        top_left = (top_left[0] + x, top_left[1] + y)
        img = cv.putText(img, text_row[1], top_left, cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 1, cv.LINE_AA)
    
    plt.imshow(img)
    plt.show()


img_cnt1 = get_text_bounds(uncropped1)
img_cnt2 = get_text_bounds(uncropped2)
img_cnt3 = get_text_bounds(uncropped3)
img_cnt4 = get_text_bounds(uncropped4)
img_cnt5 = get_text_bounds(uncropped5)


## plot individual images easyOCR
# get_text_easyOCR(img_cnt1)

## plot individual images tessOCR
get_text_tessOCR(img_cnt1)
# for img_infor in [img_cnt1, img_cnt2, img_cnt3, img_cnt4, img_cnt5]:
#     get_text_tessOCR(img_infor)


## plots images on a 2x3 grid
# _, ax = plt.subplots(2, 3, figsize=(20, 10))
# ax[0, 0].imshow(igm1)
# ax[0, 1].imshow(igm2)
# ax[0, 2].imshow(igm3)
# ax[1, 0].imshow(igm4)
# ax[1, 1].imshow(igm5)
# plt.show()
