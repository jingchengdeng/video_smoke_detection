'''
CS512 Project
Smoke detection
Su Feng
Jingcheng Deng
'''

import cv2
import numpy as np
import math
import os
import sys
from guidedfilter import *

imgf = "data/smk3.jpg"
imgsize = 700

class Node(object):
    def __init__(self, x, y, key):
        self.x = x
        self.y = y
        self.key = key


def resizeimge(img, max):
    h = img.shape[0]
    w = img.shape[1]
    if h > max or w > max:
        scale = min(float(max)/float(h),float(max)/float(w))
        return cv2.resize(img,(int(w*scale),int(h*scale)))
    return img

def colorAnalysis(img, alpha):
    img = resizeimge(img, imgsize)
    img = cv2.GaussianBlur(img, (9, 9), 5)
    cv2.imshow('imageo', img)
    h, w, d = img.shape
    red = img[:, :, 0]
    green = img[:, :, 1]
    blue = img[:, :, 2]
    for i in range(h):
        for j in range(w):
            if abs(int(red[i][j])-int(green[i][j])) < alpha \
                    and abs(int(red[i][j])-int(blue[i][j])) < alpha \
                    and abs(int(green[i][j])-int(blue[i][j])) < alpha \
                    and red[i][j] > 100:
                img[i][j] = 255
            else:
                img[i][j] = 0
    cv2.imshow('image', img)
    cv2.waitKey(0) & 0xFF


##########################################
# Darken channel helper function
# input: image, blocksize
# return: dark Channel
##########################################

def getDarkChannel(img, blocksize):
    b, g, r = cv2.split(img)
    minRBG = cv2.min(cv2.min(r, g), b)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (blocksize, blocksize))
    darkChannel = cv2.erode(minRBG, kernel)
    return darkChannel

##########################################
# Atmospheric Light helper function
# input: image, dark channel
# return: A
##########################################

def getAtomsLight(img, darkChannel):
    [h, w] = darkChannel.shape[:2]
    imgSize = h * w
    list = []
    A = 0
    for i in range(0, h):
        for j in range (0, w):
            item = Node(i, j, darkChannel[i, j])
            list.append(item)
    list.sort(key=lambda node: node.key, reverse=True)

    for i in range(0, int(imgSize * 0.1)):
        for j in range(0, 3):
            if img[list[i].x, list[i].y, j] < A:
                continue
            elif img[list[i].x, list[i].y, j] == A:
                continue
            elif img[list[i].x, list[i].y, j] > A:
                A = img[list[i].x, list[i].y, j]
    return A

##########################################
# Transmission helper function
# input: image, dark channel
# return: A
##########################################

def transmission(img, A, blocksize):
    omega = 0.95
    imageGray = np.empty(img.shape, img.dtype)
    for i in range(3):
        imageGray[:, :, i] = img[:, :, i]/A[0, i]
    t = 1 - omega * getDarkChannel(imageGray, blocksize)
    return t

def main():
    print("main")
    img = cv2.imread(imgf)
    colorAnalysis(img, 15)
'''
    try:
        video_src = sys.argv[1]
    except IndexError:
        print('Video Pass Error')
    cap = cv2.VideoCapture(video_src)
    frame_count = 1
    while True:
        ret, frame = cap.read()
        if frame is None:
            print("Video reach end.")
            break
        # extract frames for every X frame

        if frame_count % 10 == 0:
            cv2.imwrite("test/frame%d.jpg" % frame_count, frame)

        # Press Key Q to exit
        if (cv2.waitKey(10) & 0xFF) == 'Q':
            break

        frame_count += 1
'''


if __name__ == "__main__":
    main()
