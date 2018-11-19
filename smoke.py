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
import skvideo
from guidedfilter import guided_filter
import time


imgf = "data/smk3.jpg"
m1 = "test/frame10.jpg"
m2 = "test/frame15.jpg"
imgsize = 500
MHI_DURATION = 5
DEFAULT_THRESHOLD = 150
MAX_TIME_DELTA = 3
MIN_TIME_DELTA = 2
colorth = 85

class Node(object):
    def __init__(self, x, y, key):
        self.x = x
        self.y = y
        self.key = key

def svimg(img):
    cv2.imwrite('out.jpg',img)

def resizeimge(img, max):
    h = img.shape[0]
    w = img.shape[1]
    if h > max or w > max:
        scale = min(float(max)/float(h),float(max)/float(w))
        return cv2.resize(img,(int(w*scale),int(h*scale)))
    return img

def grey(img):
    global imgsize
    img = resizeimge(img, imgsize)
    if(len(img.shape)) > 2:
        return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return img

def motion(img1, img2):
    img1 = grey(img1)
    img2 = grey(img2)
    frameDelta = cv2.absdiff(img1, img2)
    return frameDelta

def motiondp(img1, img2):
    img1 = getDP(img1)
    img2 = getDP(img2)
    frameDelta = cv2.absdiff(img1, img2)
    return frameDelta

def colorAnalysis(img, alpha):
    img = resizeimge(img, imgsize)
    img = cv2.GaussianBlur(img, (5, 5), 5)
    h, w, d = img.shape
    red = img[:, :, 0]
    green = img[:, :, 1]
    blue = img[:, :, 2]
    gimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    for i in range(h):
        for j in range(w):
            if abs(int(red[i][j])-int(green[i][j])) < alpha and abs(int(red[i][j])-int(blue[i][j])) < alpha and abs(int(green[i][j])-int(blue[i][j])) < alpha and (int(red[i][j])+int(green[i][j])+int(blue[i][j]))/3 > 100:
                gimg[i][j] = 255
            else:
                gimg[i][j] = 0
    return gimg

#def motionloop():
#    im1 = cv2.imread("test/frame180.jpg")
#    im1 = colorAnalysis(im1, colorth)
#    h, w = im1.shape
#    timestamp = 0
#    motion_history = np.zeros((h, w), np.float32)
#    for i in range(180, 660, 10):
#        im1 = cv2.imread("test/frame"+str(i)+".jpg")
#        im2 = cv2.imread("test/frame"+str(i+10)+".jpg")
#        grey = motion(im1, im2,i)
#        et, motion_mask = cv2.threshold(grey, DEFAULT_THRESHOLD, 1, cv2.THRESH_BINARY)
#        timestamp += 1
#        cv2.motempl.updateMotionHistory(motion_mask, motion_history, timestamp, MHI_DURATION)
#        mg_mask, mg_orient = cv2.motempl.calcMotionGradient( motion_history, MAX_TIME_DELTA, MIN_TIME_DELTA, apertureSize=5)
#        seg_mask, seg_bounds = cv2.motempl.segmentMotion(motion_history, timestamp, MAX_TIME_DELTA)
#        vis = np.uint8(np.clip((motion_history-(timestamp-MHI_DURATION)) / MHI_DURATION, 0, 1)*255)
#        cv2.imshow('motempl', vis)
#        if k == 27:
#            cv2.destroyAllWindows()
#            exit()

def mhi(fn, st, n, intv):
    im1 = cv2.imread(fn + "/frame" +str(st)+ ".jpg")
    im1 = grey(im1)
    h, w = im1.shape
    timestamp = 0
    motion_history = np.zeros((h, w), np.float32)
    for i in range(st, st+n*intv+1, intv):
        im1 = cv2.imread(fn + "/frame"+str(st)+".jpg")
        im2 = cv2.imread(fn + "/frame"+str(i+intv)+".jpg")
        gry = motion(im1, im2)
        et, motion_mask = cv2.threshold(gry, DEFAULT_THRESHOLD, 1, cv2.THRESH_BINARY)
        timestamp += 1
        cv2.motempl.updateMotionHistory(motion_mask, motion_history, timestamp, MHI_DURATION)
        mg_mask, mg_orient = cv2.motempl.calcMotionGradient( motion_history, MAX_TIME_DELTA, MIN_TIME_DELTA, apertureSize=5)
        seg_mask, seg_bounds = cv2.motempl.segmentMotion(motion_history, timestamp, MAX_TIME_DELTA)
        vis = np.uint8(np.clip((motion_history-(timestamp-MHI_DURATION)) / MHI_DURATION, 0, 1)*255)
    return vis



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
    #print(imageGray)
    #print(A)
    # print(getDarkChannel(imageGray, blocksize))
    t = 1 - omega * getDarkChannel(imageGray, blocksize)
    # print(t)
    t[t<0.1]= 0.1
    # normI = (img - img.min()) / (img.max() - img.min())
    # t = guided_filter(normI, t, 40, 0.0001)
    # print(t)
    return t

def getDP(image):
    image = resizeimge(image, imgsize)
    I = image.astype('float64') / 255
    darkChannel = getDarkChannel(I, 15)
    A = getAtomsLight(I, darkChannel)
    t = transmission(I, A, 15, image)
    #print('Done!')
    
    h, w, d = image.shape
    gimg = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    for i in range(h):
        for j in range(w):
            if t[i][j] < 0.4:
                gimg[i][j] = 255
            else:
                gimg[i][j] = 0
    return gimg, t
#print(t.shape)
#print(image.shape)
#cv2.imshow('DP', gimg)

def stack(img, img2, img3):
    out = img.copy()
    h,w = img.shape
    for i in range(h):
        for j in range(w):
            if img[i][j] > 75 and img2[i][j] > 75 and img3[i][j] > 75:
                out[i][j] = 255
            else:
                out[i][j] = 0
    return out

# input: original image,
def drawmask(img, mask, n=3):
    out = img.copy()
    overlay = img.copy()
    h,w = mask.shape
    for i in range(2,h-2,2*n):
        for j in range(2,w-2,2*n):
            if mask[i][j] == 255:
                cv2.rectangle(overlay, (j-n, i-n), (j+n, i+n),(0, 0, 255), -1)
    cv2.addWeighted(overlay, 0.5, out, 0.5,0, out)
    return out

def productVideo(h,w):
    try:
         video_src = sys.argv[1]
    except IndexError:
         print('Video Pass Error')
    cap = cv2.VideoCapture(video_src)
    fps = 15
    capSize = (281, 500)
    frame_count = 1
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter('output.mp4', fourcc, 15.0, (1280, 720), True)
    while True:
        print(frame_count)
        ret, frame = cap.read()
        if frame is None:
            print("Video reach end.")
            break

        # frame = getDP(frame)
        # frame_width = int(frame.get(3))
        # frame_height = int(frame.get(4))
        frame = cv2.resize(frame, (1280,720))
        print(frame.shape)
        h,w,d = frame.shape
        print(h)
        print(w)
        out.write(frame)
        frame_count += 1
        if frame_count == 91:
            out.release()
            break

def extract_frames(fn):
    try:
        video_src = fn
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
        
        if frame_count % 5 == 0:
            cv2.imwrite("test2/frame%d.jpg" % frame_count, frame)
            cv2.imshow('ext', frame)
        
        # Press Key Q to exit
        if (cv2.waitKey(10) & 0xFF) == 'Q':
            break
        frame_count += 1

def main():
    print("main")
    # motionloop()

    img = cv2.imread("test2/frame255.jpg")
#    img = resizeimge(img, imgsize)
#    h,w,d = img.shape
#    img1 = colorAnalysis(img,colorth)
#    #cv2.imshow('grey', img1)
    img2, t = getDP(img)
#img3 = mhi("test2", 255, 5, 5)
#cv2.imshow('mhi', img3)
#    final = stack(img1,img2,img3)
#    cv2.imshow('final', final)
#    ovl = drawmask(img, final)
#    cv2.imshow('overlay', ovl)


#extract_frames("videos/simple_smoke.mp4")
    cv2.waitKey(0)


#
#

if __name__ == "__main__":
    main()
