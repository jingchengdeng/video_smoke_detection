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

imgf = "data/smk1.jpg"
imgsize = 700

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
            if abs(int(red[i][j])-int(green[i][j])) < alpha and abs(int(red[i][j])-int(blue[i][j])) < alpha and abs(int(green[i][j])-int(blue[i][j])) < alpha and red[i][j] > 100:
                img[i][j] = 255
            else:
                img[i][j] = 0
    cv2.imshow('image', img)
    cv2.waitKey(0) & 0xFF


def main():
    print("main")
    img = cv2.imread(imgf)
    colorAnalysis(img, 15)







if __name__ == "__main__":
    main()
