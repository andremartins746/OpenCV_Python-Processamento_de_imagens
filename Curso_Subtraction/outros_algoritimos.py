from random import randint
import cv2
import sys
import numpy as np


TEXT_COLOR = (randint(0, 255), randint(0, 255), randint(0, 255))
BORDER_COLOR = (randint(0, 255), randint(0, 255), randint(0, 255))
FONT = cv2.FONT_HERSHEY_SIMPLEX
VIDEO_SOURCE = 'videos/Traffic_4.mp4'

# print(TEXT_COLOR, BORDER_COLOR)

BGS_TYPES = ['GMC', 'MOG2', 'MOG', 'KNN', 'CNT']


def getKernel(KERNEL_TYPE):
    if KERNEL_TYPE == 'dilation':
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    if KERNEL_TYPE == 'opening':
        kernel = np.ones((3,3), np.unint8)

    if KERNEL_TYPE == 'closing':
        kernel = np.ones((3, 3), np.unint8)

    return kernel

#print(getKernel('dilation'))

def getFilter (img, filter):
    if filter == 'closing':
        return cv2.morphologyEx(img,cv2.MORPH_CLOSE, getKernel('closing'), iterations=2)

    if filter == 'opening':
        return cv2.morphologyEx(img, cv2.MORPH_OPEN, getKernel('opening'), iterations=2)

    if filter == 'dilation':
        return cv2.morphologyEx(img, getKernel('dilation'), iterations=2)

    if filter == 'conbine':
        closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, getKernel('closing'), iterations=2)
        opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, getKernel('opening'), iterations=2)
        dilation = cv2.dilate(opening, getKernel('dilation'), iterations=2)
        return dilation