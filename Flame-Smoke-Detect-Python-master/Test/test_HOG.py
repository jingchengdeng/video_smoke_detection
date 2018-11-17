# *^_^* coding:utf-8 *^_^*
__author__ = 'stone'
__date__ = '15-11-11'

import cv2
import numpy as np

img = cv2.imread('../medias/smoke/positive/shot0052.png')
winSize = (640, 480)
padding = (8, 8)
blockStride = (8, 8)

descriptor = cv2.HOGDescriptor()
hog = descriptor.compute(img, blockStride, padding)
print type(hog)

np.savetxt('hog.txt', hog)