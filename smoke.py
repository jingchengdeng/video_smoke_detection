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

def main():
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


if __name__ == "__main__":
    main()
