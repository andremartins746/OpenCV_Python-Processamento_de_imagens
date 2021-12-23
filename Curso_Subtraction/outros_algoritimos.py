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
    
    
    
    def getBGSubtractor(BGS_TYPE):
        
             if BGS_TYPE == 'GMG':
                    return cv2.bgsegm.createBacgroundSubtractorGMG(initiationFrames = 120, decisionThreshold= 0.8)

             if BGS_TYPE == 'MOG':
                 return cv2.bgsegm.createBacgroundSubtractorMOG(history= 200, nmixtures = 5, backgroundRation= 0.7, noiseSigma = 0)
                
             if BGS_TYPE == 'MOG2':
                 return cv2.createBacgroundSubtractorMOG2(history= 500, detectShadows= True, varThresHold=100 )

             if BGS_TYPE == 'KNN':
                 return cv2.createBacgroundSubtractorKNN(history=500, dist2Thresthold=400, detectShadow= True)

             if BGS_TYPE == 'CNT':
                 return cv2.createBacgroundSubtractorCNT(minPixelStability=15, useHistory=True, maxPixelStability = 15 * 60, isParallel=True)
                
            print('Detector invalido')
            sys.exit(1)
