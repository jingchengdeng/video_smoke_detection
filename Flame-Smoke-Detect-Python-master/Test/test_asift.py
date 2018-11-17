#!/usr/bin/env python

'''
根据opencv3.1.0/sample/python/asift.py修改
目的：根据上下帧的对应点检测实现DMHI，检测目标运动方向。未完成。
思路：先提取运动区域，然后提取上下帧的对应点，再对对应点生成DMHI。
完成度：实现了上下帧对应点的检测。
缺陷：对于运动幅度较小的目标，对应点难以确认。

Affine invariant feature-based image matching sample.

This sample is similar to find_obj.py, but uses the affine transformation
space sampling technique, called ASIFT [1]. While the original implementation
is based on SIFT, you can try to use SURF or ORB detectors instead. Homography RANSAC
is used to reject outliers. Threading is used for faster affine sampling.

[1] http://www.ipol.im/pub/algo/my_affine_sift/
'''

# Python 2/3 compatibility
from __future__ import print_function

import numpy as np
import cv2

# built-in modules
import itertools as it
from multiprocessing.pool import ThreadPool

# local modules
from common import Timer

COUNT = 0


def filter_matches(kp1, kp2, matches, ratio=0.75):
    mkp1, mkp2 = [], []
    for m in matches:
        if len(m) == 2 and m[0].distance < m[1].distance * ratio:
            m = m[0]
            mkp1.append(kp1[m.queryIdx])
            mkp2.append(kp2[m.trainIdx])
    p1 = np.float32([kp.pt for kp in mkp1])
    p2 = np.float32([kp.pt for kp in mkp2])
    kp_pairs = zip(mkp1, mkp2)
    return p1, p2, kp_pairs


def explore_match(win, img1, img2, kp_pairs, status=None, H=None):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    vis = np.zeros((max(h1, h2), w1 + w2), np.uint8)
    vis[:h1, :w1] = img1
    vis[:h2, w1:w1 + w2] = img2
    vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)

    if H is not None:
        corners = np.float32([[0, 0], [w1, 0], [w1, h1], [0, h1]])
        corners = np.int32(cv2.perspectiveTransform(corners.reshape(1, -1, 2), H).reshape(-1, 2) + (w1, 0))
        cv2.polylines(vis, [corners], True, (255, 255, 255))

    if status is None:
        status = np.ones(len(kp_pairs), np.bool_)
    p1, p2 = [], []  # python 2 / python 3 change of zip unpacking
    for kpp in kp_pairs:
        p1.append(np.int32(kpp[0].pt))
        p2.append(np.int32(np.array(kpp[1].pt) + [w1, 0]))

    green = (0, 255, 0)
    red = (0, 0, 255)
    white = (255, 255, 255)
    kp_color = (51, 103, 236)
    for (x1, y1), (x2, y2), inlier in zip(p1, p2, status):
        if inlier:
            col = green
            cv2.circle(vis, (x1, y1), 2, col, -1)
            cv2.circle(vis, (x2, y2), 2, col, -1)
        else:
            col = red
            r = 2
            thickness = 3
            cv2.line(vis, (x1 - r, y1 - r), (x1 + r, y1 + r), col, thickness)
            cv2.line(vis, (x1 - r, y1 + r), (x1 + r, y1 - r), col, thickness)
            cv2.line(vis, (x2 - r, y2 - r), (x2 + r, y2 + r), col, thickness)
            cv2.line(vis, (x2 - r, y2 + r), (x2 + r, y2 - r), col, thickness)
    vis0 = vis.copy()
    for (x1, y1), (x2, y2), inlier in zip(p1, p2, status):
        if inlier:
            cv2.line(vis, (x1, y1), (x2, y2), green)

    cv2.imshow(win, vis)


def affine_skew(tilt, phi, img, mask=None):
    '''
    affine_skew(tilt, phi, img, mask=None) -> skew_img, skew_mask, Ai

    Ai - is an affine transform matrix from skew_img to img
    '''
    h, w = img.shape[:2]
    if mask is None:
        mask = np.zeros((h, w), np.uint8)
        mask[:] = 255
    A = np.float32([[1, 0, 0], [0, 1, 0]])
    if phi != 0.0:
        phi = np.deg2rad(phi)
        s, c = np.sin(phi), np.cos(phi)
        A = np.float32([[c, -s], [s, c]])
        corners = [[0, 0], [w, 0], [w, h], [0, h]]
        tcorners = np.int32(np.dot(corners, A.T))
        x, y, w, h = cv2.boundingRect(tcorners.reshape(1, -1, 2))
        A = np.hstack([A, [[-x], [-y]]])
        img = cv2.warpAffine(img, A, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    if tilt != 1.0:
        s = 0.8 * np.sqrt(tilt * tilt - 1)
        img = cv2.GaussianBlur(img, (0, 0), sigmaX=s, sigmaY=0.01)
        img = cv2.resize(img, (0, 0), fx=1.0 / tilt, fy=1.0, interpolation=cv2.INTER_NEAREST)
        A[0] /= tilt
    if phi != 0.0 or tilt != 1.0:
        h, w = img.shape[:2]
        mask = cv2.warpAffine(mask, A, (w, h), flags=cv2.INTER_NEAREST)
    Ai = cv2.invertAffineTransform(A)
    return img, mask, Ai


def affine_detect(detector, img, mask=None, pool=None):
    '''
    affine_detect(detector, img, mask=None, pool=None) -> keypoints, descrs

    Apply a set of affine transormations to the image, detect keypoints and
    reproject them into initial image coordinates.
    See http://www.ipol.im/pub/algo/my_affine_sift/ for the details.

    ThreadPool object may be passed to speedup the computation.
    '''
    params = [(1.0, 0.0)]
    for t in 2 ** (0.5 * np.arange(1, 6)):
        for phi in np.arange(0, 180, 72.0 / t):
            params.append((t, phi))

    def f(p):
        t, phi = p
        timg, tmask, Ai = affine_skew(t, phi, img)
        keypoints, descrs = detector.detectAndCompute(timg, tmask)
        for kp in keypoints:
            x, y = kp.pt
            kp.pt = tuple(np.dot(Ai, (x, y, 1)))
        if descrs is None:
            descrs = []
        return keypoints, descrs

    keypoints, descrs = [], []
    if pool is None:
        ires = it.imap(f, params)
    else:
        ires = pool.imap(f, params)

    for i, (k, d) in enumerate(ires):
        print('affine sampling: %d / %d\r' % (i + 1, len(params)), end='')
        keypoints.extend(k)
        descrs.extend(d)

    print()
    return keypoints, np.array(descrs)


if __name__ == '__main__':
    print(__doc__)
    video_src = '/home/stone/Code/FlameSmokeDetect/medias/videos/CTC_FG.028_9_320x240.avi'

    cap = cv2.VideoCapture(video_src)
    fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows=False)
    ret, frame_old = cap.read()
    frame_gray_old = cv2.cvtColor(frame_old, cv2.COLOR_BGR2GRAY)

    cap = cv2.VideoCapture(video_src)
    while True:
        ret, frame = cap.read()
        if frame is None:
            print("Video playback is completed")
            break
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        fmask = fgbg.apply(frame)
        dilate_kernel = np.ones((8, 8), np.uint8)
        fmask = cv2.medianBlur(fmask, 5)
        fmask_dilate = cv2.dilate(fmask, dilate_kernel, iterations=2)
        frame_and = cv2.bitwise_and(frame_gray, frame_gray, mask=fmask_dilate)

        fmask_old = fgbg.apply(frame_old)
        fmask_old = cv2.medianBlur(fmask_old, 5)
        fmask_dilate_old = cv2.dilate(fmask_old, dilate_kernel, iterations=2)
        not_fmask_dilate = np.ones_like(fmask_dilate)
        fmask_dilate_old = cv2.bitwise_not(fmask_dilate, fmask_dilate, mask=not_fmask_dilate)
        frame_and_old = cv2.bitwise_not(frame_gray_old, frame_gray_old, mask=fmask_dilate_old)

        detector = cv2.xfeatures2d.SIFT_create()
        norm = cv2.NORM_L2
        matcher = cv2.BFMatcher(norm)

        pool = ThreadPool(processes=cv2.getNumberOfCPUs())
        kp1, desc1 = affine_detect(detector, frame_gray, pool=pool)
        kp2, desc2 = affine_detect(detector, frame_gray_old, pool=pool)
        print('img1 - %d features, img2 - %d features' % (len(kp1), len(kp2)))


        def match_and_draw(win):
            with Timer('matching'):
                raw_matches = matcher.knnMatch(desc1, trainDescriptors=desc2, k=2)  # 2
            p1, p2, kp_pairs = filter_matches(kp1, kp2, raw_matches)
            if len(p1) >= 4:
                H, status = cv2.findHomography(p1, p2, cv2.RANSAC, 5.0)
                # print('%d / %d  inliers/matched' % (np.sum(status), len(status)))
                # do not draw outliers (there will be a lot of them)
                try:
                    kp_pairs = [kpp for kpp, flag in zip(kp_pairs, status) if flag]
                except:
                    pass
            else:
                H, status = None, None
                print('%d matches found, not enough for homography estimation' % len(p1))

            explore_match(win, frame_and, frame_and_old, kp_pairs, None, H)


        match_and_draw('affine find_obj')
        frame_gray_old = frame_gray
        frame_old = frame

        COUNT += 1
        if cv2.waitKey(10) & 0xFF == 27:
            break
    cv2.destroyAllWindows()
    cap.release()
