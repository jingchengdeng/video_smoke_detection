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
import time

imgf = "data/smk3.jpg"
m1 = "test/frame180.jpg"
m2 = "test/frame190.jpg"
imgsize = 700
MHI_DURATION = 5
DEFAULT_THRESHOLD = 68
MAX_TIME_DELTA = 3
MIN_TIME_DELTA = 2

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

def main():
    print("main")
    motionloop()

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
