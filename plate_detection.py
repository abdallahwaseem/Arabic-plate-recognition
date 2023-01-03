# import the necessary packages
from skimage.segmentation import clear_border
import commonfunctions as cf
import math
import os
import numpy as np
import imutils
import cv2
from skimage.measure import find_contours
from skimage.draw import rectangle
from skimage.morphology import binary_erosion, binary_dilation
from skimage.feature import canny
from skimage.filters import threshold_otsu, sobel
from skimage.transform import rotate
from skimage.transform import (hough_line, hough_line_peaks)
from scipy.stats import mode
from joblib import load


def plate_detection_morphological_operations(thresh):
    thresh = cv2.erode(thresh, None, iterations=3)
    thresh = cv2.dilate(thresh, None, iterations=8)
    thresh = cv2.erode(thresh, None, iterations=1)
    thresh = cv2.dilate(thresh, None, iterations=7)
    thresh = cv2.erode(thresh, None, iterations=1)
    return thresh


def post_processing_morphological_operations(thresh, rectKern):
    thresh = cv2.dilate(thresh, rectKern, iterations=8)
    thresh = cv2.dilate(thresh, None, iterations=8)
    thresh = cv2.erode(thresh, None, iterations=2)
    return thresh


def skew_angle_hough_transform(image):
    # convert to edges
    edges = canny(image)
    # Classic straight-line Hough transform between 0.1 - 180 degrees.
    h, theta, d = hough_line(edges)
    # find line peaks and angles
    accum, angles, dists = hough_line_peaks(h, theta, d)
    # print(angles)
    # round the angles to 2 decimal places and find the most common angle.
    most_common_angle = mode(np.around(angles, decimals=2))[0]
    # print(most_common_angle)
    # convert the angle to degree for rotation.
    skew_angle = np.rad2deg(most_common_angle)
    if skew_angle < 0:
        skew_angle = skew_angle+90
    else:
        skew_angle = skew_angle-90

    return skew_angle


def skewRotation(plate):
    plateEdges = canny(plate)
    angle = skew_angle_hough_transform(plate)
    if len(angle) == 0:
        angle = 0
    else:
        angle = angle[0]
    rotated_image = rotate(plate, angle)
    return rotated_image


def plate_detection(car_img, post_processing=False):
    # 1- Resize image
    if not post_processing:
        car_img = cv2.resize(car_img, (800, 600))
    # 2- Convert to gray
    gray = cv2.cvtColor(car_img, cv2.COLOR_BGR2GRAY)
    rectKern = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))

    # 3- Perform a blackhat morphological operation
    # It will allow us to reveal dark regions (i.e., text) on light backgrounds (i.e., the license plate itself)
    # black-hat transform =  closing - input image
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rectKern)
    # 4- Applying closing using square structuring element
    # To find regions in the image that are light and may contain license plate characters:
    squareStrucElement = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    light = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, rectKern)
    # 5- binary thresholding, if pixel intensity is greater than threshold set it to 255, else set it to 0
    light = cv2.threshold(
        light, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    # 6- compute the Sobel gradient representation of the blackhat
    # image in the x-direction and then scale the result back to the range [0, 255]
    gradX = cv2.Sobel(blackhat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
    gradX = np.absolute(gradX)
    (minVal, maxVal) = (np.min(gradX), np.max(gradX))
    gradX = 255 * ((gradX - minVal) / (maxVal - minVal))
    gradX = gradX.astype("uint8")
    # show the gradient image
    # cf.show_images([gradX], ["gradX"])

    # 7- blur the gradient representation, applying a closing operation, and threshold the image using Otsu's method
    gradX = cv2.GaussianBlur(gradX, (5, 5), 0)
    gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKern)

    thresh = cv2.threshold(
        gradX, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    # 8- perform a series of erosions and dilations to clean up the thresholded image
    if not post_processing:
        result_img = plate_detection_morphological_operations(thresh)
    else:
        result_img = post_processing_morphological_operations(thresh, rectKern)

    # 9- take the bitwise AND between the threshold result and the light regions of the image
    result_img = cv2.bitwise_and(result_img, result_img, mask=light)

    if not post_processing:
        result_img = cv2.dilate(result_img, None, iterations=8)
        result_img = cv2.erode(result_img, None, iterations=1)

    # 10- find contours in the thresholded image
    # sort them by their size in descending order, keeping only the largest ones
    cnts = cv2.findContours(
        result_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:30]

    # initialize the license plate contour
    licensePlate = None
    # These are the car contours below
    # 11-loop over the license plate candidate contours
    for c in cnts:
        (x, y, w, h) = cv2.boundingRect(c)
        lpCnt = c
        # If it was post processing compute some checks
        if post_processing:
            if(w > 60):
                licensePlate = gray[y:y + h, x:x + w]
                rotated_plate = skewRotation(licensePlate)
                break
        else:
            # If it was plate detection compute some checks
            if w > 300 or h > 140:
                continue
            if(w > 1.3*h and w < 3.5*h):
                licensePlate = gray[y:y + h, x:x + w]
                break
    return licensePlate
