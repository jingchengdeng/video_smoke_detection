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
from guidedfilter import guided_filter
import time

imgf = "data/smk3.jpg"
m1 = "test/frame180.jpg"
m2 = "test/frame190.jpg"
imgsize = 700
MHI_DURATION = 5
DEFAULT_THRESHOLD = 68
MAX_TIME_DELTA = 3
MIN_TIME_DELTA = 2

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

def grey(img):
    if(len(img.shape)) > 2:
        return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return img

def motion(img1, img2):
    img1 = colorAnalysis(img1, imgsize)
    img2 = colorAnalysis(img2, imgsize)
    frameDelta = cv2.absdiff(img1, img2)
    return frameDelta

def colorAnalysis(img, alpha):
    img = resizeimge(img, imgsize)
    img = cv2.GaussianBlur(img, (9, 9), 5)
    h, w, d = img.shape
    red = img[:, :, 0]
    green = img[:, :, 1]
    blue = img[:, :, 2]
    gimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    for i in range(h):
        for j in range(w):
            if abs(int(red[i][j])-int(green[i][j])) < alpha and abs(int(red[i][j])-int(blue[i][j])) < alpha and abs(int(green[i][j])-int(blue[i][j])) < alpha and red[i][j] > 100:
                gimg[i][j] = 255
            else:
                gimg[i][j] = 0
    return gimg

def motionloop():
    im1 = cv2.imread("test/frame180.jpg")
    im1 = colorAnalysis(im1, 15)
    h, w = im1.shape
    timestamp = 0
    motion_history = np.zeros((h, w), np.float32)
    for i in range(180, 660, 10):
        im1 = cv2.imread("test/frame"+str(i)+".jpg")
        im2 = cv2.imread("test/frame"+str(i+10)+".jpg")
        grey = motion(im1, im2)
        et, motion_mask = cv2.threshold(grey, DEFAULT_THRESHOLD, 1, cv2.THRESH_BINARY)
        timestamp += 1
        cv2.motempl.updateMotionHistory(motion_mask, motion_history, timestamp, MHI_DURATION)
        mg_mask, mg_orient = cv2.motempl.calcMotionGradient( motion_history, MAX_TIME_DELTA, MIN_TIME_DELTA, apertureSize=5)
        seg_mask, seg_bounds = cv2.motempl.segmentMotion(motion_history, timestamp, MAX_TIME_DELTA)
        vis = np.uint8(np.clip((motion_history-(timestamp-MHI_DURATION)) / MHI_DURATION, 0, 1)*255)
        cv2.imshow('motempl', vis)
        k = cv2.waitKey(0) & 0xFF
        if k == 27:
            cv2.destroyAllWindows()
            exit()

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

    pixl = int(max(math.floor(imgSize/1000),1))
    darkReshape = darkChannel.reshape(imgSize, 1)
    imageReshape = img.reshape(imgSize, 3)
    I = darkReshape.argsort()
    I = I[imgSize-pixl::]

    at = np.zeros([1, 3])
    for i in range(1, pixl):
        at = at +imageReshape[I[i]]
    A = at/pixl
    return A

##########################################
# Transmission helper function
# input: image, dark channel
# return: A
##########################################

def transmission(img, A, blocksize, ori):
    omega = 0.95
    imageGray = np.empty(img.shape, img.dtype)
    # imageGray = np.min(img, axis=2)
    # print(A)
    for i in range(3):
        imageGray[:, :, i] = img[:, :, i]/A[0, i]
    print(imageGray)
    print(A)
    # print(getDarkChannel(imageGray, blocksize))
    t = 1 - omega * getDarkChannel(imageGray, blocksize)
    # print(t)
    t[t<0.1]= 0.1
    normI = (img - img.min()) / (img.max() - img.min())
    t = guided_filter(normI, t, 40, 0.0001)
    # print(t)
    return t

def main():
    print("main")
    # motionloop()
    image = cv2.imread("test/frame10.jpg")
    image = resizeimge(image, 500)
    I = image.astype('float64') / 255
    darkChannel = getDarkChannel(I, 15)
    A = getAtomsLight(I, darkChannel)
    t = transmission(I, A, 15, image)
    print('Done!')

    h, w, d = image.shape
    gimg = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    for i in range(h):
        for j in range(w):
            if t[i][j] < 0.4:
                gimg[i][j] = 255
            else:
                gimg[i][j] = 0
    print(t.shape)
    print(image.shape)
    cv2.imshow('DP', gimg)
    cv2.waitKey(0)

    # print(t)


#     try:
#         video_src = sys.argv[1]
#     except IndexError:
#         print('Video Pass Error')
#     cap = cv2.VideoCapture(video_src)
#     frame_count = 1
#     while True:
#         ret, frame = cap.read()
#         if frame is None:
#             print("Video reach end.")
#             break
#         # extract frames for every X frame
#
#         if frame_count % 5 == 0:
#             cv2.imwrite("test/frame%d.jpg" % frame_count, frame)
#
#         # Press Key Q to exit
#         if (cv2.waitKey(10) & 0xFF) == 'Q':
#             break
#
#         frame_count += 1
#
#

if __name__ == "__main__":
    main()
